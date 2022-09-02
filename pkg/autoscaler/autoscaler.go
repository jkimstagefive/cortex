/*
Copyright 2022 Cortex Labs, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package autoscaler

import (
	"fmt"
	"math"
	"time"

	"github.com/cortexlabs/cortex/pkg/lib/cron"
	"github.com/cortexlabs/cortex/pkg/lib/errors"
	libmath "github.com/cortexlabs/cortex/pkg/lib/math"
	"github.com/cortexlabs/cortex/pkg/lib/pointer"
	"github.com/cortexlabs/cortex/pkg/lib/telemetry"
	libtime "github.com/cortexlabs/cortex/pkg/lib/time"
	"github.com/cortexlabs/cortex/pkg/types/spec"
	"github.com/cortexlabs/cortex/pkg/types/userconfig"
	"go.uber.org/zap"
	redis "github.com/go-redis/redis"
        sstrings "strings"
        "strconv"
)

func initializeRedisClient() (*redis.Client, error) {
        client := redis.NewClient(&redis.Options{
        Addr:     "23.23.220.207:6379", // 접근 url 및 port
        Password: "redisscheduler",               // password ""값은 없다는 뜻
        DB:       0,                // 기본 DB 사용
        })

        _, err := client.Ping().Result()
        return client, err
}

const (
	_prometheusQueryTimeoutSeconds = 10
)

type Scaler interface {
	Scale(apiName string, request int32) error
	GetInFlightRequests(apiName string, window time.Duration) (*float64, error)
	GetAutoscalingSpec(apiName string) (*userconfig.Autoscaling, error)
	CurrentRequestedReplicas(apiName string) (int32, error)
}

type Autoscaler struct {
	logger  *zap.SugaredLogger
	crons   map[string]cron.Cron
	scalers map[userconfig.Kind]Scaler
	recs    map[string]*recommendations
}

func New(logger *zap.SugaredLogger) *Autoscaler {
	return &Autoscaler{
		logger:  logger,
		crons:   make(map[string]cron.Cron),
		scalers: make(map[userconfig.Kind]Scaler),
		recs:    make(map[string]*recommendations),
	}
}

func (a *Autoscaler) AddScaler(scaler Scaler, kind userconfig.Kind) {
	a.scalers[kind] = scaler
}

func (a *Autoscaler) Awaken(api userconfig.Resource) error {
	scaler, ok := a.scalers[api.Kind]
	if !ok {
		return errors.ErrorUnexpected(
			fmt.Sprintf("autoscaler does not have a scaler for the %s kind", api.Kind),
		)
	}

	log := a.logger.With(
		zap.String("apiName", api.Name),
		zap.String("apiKind", api.Kind.String()),
	)

	currentRequestedReplicas, err := scaler.CurrentRequestedReplicas(api.Name)
	if err != nil {
		return errors.Wrap(err, "failed to get current replicas")
	}

	if currentRequestedReplicas > 0 {
		return nil
	}

	log.Infof("autoscaling awake event")
	if err := scaler.Scale(api.Name, 1); err != nil {
		return errors.Wrap(err, "failed to scale api to one")
	}

	a.recs[api.Name].add(1)

	return nil
}

func (a *Autoscaler) AddAPI(api userconfig.Resource) error {
	if _, ok := a.crons[api.Name]; ok {
		return nil
	}

	errorHandler := func(err error) {
		log := a.logger.With(
			zap.String("apiName", api.Name),
			zap.String("apiKind", api.Kind.String()),
		)

		log.Error(err)
		telemetry.Error(err)
	}

	autoscaleFn, err := a.autoscaleFn(api)
	if err != nil {
		return err
	}

	a.crons[api.Name] = cron.Run(autoscaleFn, errorHandler, spec.AutoscalingTickInterval)

	return nil
}

func (a *Autoscaler) RemoveAPI(api userconfig.Resource) {
	log := a.logger.With(
		zap.String("apiName", api.Name),
		zap.String("apiKind", api.Kind.String()),
	)

	if autoscalerCron, ok := a.crons[api.Name]; ok {
		autoscalerCron.Cancel()
		delete(a.crons, api.Name)
	}

	delete(a.recs, api.Name)
	log.Info("autoscaler stop")
}

func (a *Autoscaler) Stop() {
	for apiName, apiCron := range a.crons {
		apiCron.Cancel()
		delete(a.crons, apiName)
	}
}

func (a *Autoscaler) autoscaleFn(api userconfig.Resource) (func() error, error) {
	log := a.logger.With(
		zap.String("apiName", api.Name),
		zap.String("apiKind", api.Kind.String()),
	)

	scaler, ok := a.scalers[api.Kind]
	if !ok {
		return nil, errors.ErrorUnexpected(
			fmt.Sprintf("autoscaler does not have a scaler for the %s kind", api.Kind),
		)
	}

	log.Info("autoscaler init")

	var startTime time.Time
	a.recs[api.Name] = newRecommendations()

	return func() error {
		autoscalingSpec, err := scaler.GetAutoscalingSpec(api.Name)
		if err != nil {
			return errors.Wrap(err, "failed to get autoscaling spec")
		}

		currentRequestedReplicas, err := scaler.CurrentRequestedReplicas(api.Name)
		if err != nil {
			return errors.Wrap(err, "failed to get current replicas")
		}

		if startTime.IsZero() {
			startTime = time.Now()
		}

		avgInFlight, err := scaler.GetInFlightRequests(api.Name, autoscalingSpec.Window)
		if err != nil {
			return errors.Wrap(err, "failed to get in-flight requests")
		}
		if avgInFlight == nil {
			log.Debug("autoscaler tick: metrics not available yet")
			return nil
		}

		rawRecommendation := *avgInFlight / *autoscalingSpec.TargetInFlight
		recommendation := int32(math.Ceil(rawRecommendation))

		if rawRecommendation < float64(currentRequestedReplicas) && rawRecommendation > float64(currentRequestedReplicas)*(1-autoscalingSpec.DownscaleTolerance) {
			recommendation = currentRequestedReplicas
		}

		if rawRecommendation > float64(currentRequestedReplicas) && rawRecommendation < float64(currentRequestedReplicas)*(1+autoscalingSpec.UpscaleTolerance) {
			recommendation = currentRequestedReplicas
		}

		// always allow subtraction of 1
		downscaleFactorFloor := libmath.MinInt32(currentRequestedReplicas-1, int32(math.Ceil(float64(currentRequestedReplicas)*autoscalingSpec.MaxDownscaleFactor)))
		if recommendation < downscaleFactorFloor {
			recommendation = downscaleFactorFloor
		}

		// always allow addition of 1
		upscaleFactorCeil := libmath.MaxInt32(currentRequestedReplicas+1, int32(math.Ceil(float64(currentRequestedReplicas)*autoscalingSpec.MaxUpscaleFactor)))
		if recommendation > upscaleFactorCeil {
			recommendation = upscaleFactorCeil
		}

		if recommendation < autoscalingSpec.MinReplicas {
			recommendation = autoscalingSpec.MinReplicas
		}

		if recommendation > autoscalingSpec.MaxReplicas {
			recommendation = autoscalingSpec.MaxReplicas
		}

		recs := a.recs[api.Name]

		// Rule of thumb: any modifications that don't consider historical recommendations should be performed before
		// recording the recommendation, any modifications that use historical recommendations should be performed after
		recs.add(recommendation)

		// This is just for garbage collection
		recs.deleteOlderThan(libtime.MaxDuration(autoscalingSpec.DownscaleStabilizationPeriod, autoscalingSpec.UpscaleStabilizationPeriod))

		request := recommendation
		var downscaleStabilizationFloor *int32
		var upscaleStabilizationCeil *int32

		if request < currentRequestedReplicas {
			downscaleStabilizationFloor = recs.maxSince(autoscalingSpec.DownscaleStabilizationPeriod)
			if downscaleStabilizationFloor != nil {
				downscaleStabilizationFloor = pointer.Int32(libmath.MinInt32(*downscaleStabilizationFloor, currentRequestedReplicas))
			}
			if time.Since(startTime) < autoscalingSpec.DownscaleStabilizationPeriod {
				request = currentRequestedReplicas
			} else if downscaleStabilizationFloor != nil && request < *downscaleStabilizationFloor {
				request = *downscaleStabilizationFloor
			}
		}
		if request > currentRequestedReplicas {
			upscaleStabilizationCeil = recs.minSince(autoscalingSpec.UpscaleStabilizationPeriod)
			if upscaleStabilizationCeil != nil {
				upscaleStabilizationCeil = pointer.Int32(libmath.MaxInt32(*upscaleStabilizationCeil, currentRequestedReplicas))
			}
			if time.Since(startTime) < autoscalingSpec.UpscaleStabilizationPeriod {
				request = currentRequestedReplicas
			} else if upscaleStabilizationCeil != nil && request > *upscaleStabilizationCeil {
				request = *upscaleStabilizationCeil
			}
		}

		log.Debugw("autoscaler tick",
			"autoscaling", map[string]interface{}{
				"avg_in_flight":                  *avgInFlight,
				"target_in_flight":               *autoscalingSpec.TargetInFlight,
				"raw_recommendation":             rawRecommendation,
				"current_replicas":               currentRequestedReplicas,
				"downscale_tolerance":            autoscalingSpec.DownscaleTolerance,
				"upscale_tolerance":              autoscalingSpec.UpscaleTolerance,
				"max_downscale_factor":           autoscalingSpec.MaxDownscaleFactor,
				"downscale_factor_floor":         downscaleFactorFloor,
				"max_upscale_factor":             autoscalingSpec.MaxUpscaleFactor,
				"upscale_factor_ceil":            upscaleFactorCeil,
				"min_replicas":                   autoscalingSpec.MinReplicas,
				"max_replicas":                   autoscalingSpec.MaxReplicas,
				"recommendation":                 recommendation,
				"downscale_stabilization_period": autoscalingSpec.DownscaleStabilizationPeriod.Seconds(),
				"downscale_stabilization_floor":  downscaleStabilizationFloor,
				"upscale_stabilization_period":   autoscalingSpec.UpscaleStabilizationPeriod.Seconds(),
				"upscale_stabilization_ceil":     upscaleStabilizationCeil,
				"request":                        request,
			},
		)

		// redis connection
                client, err := initializeRedisClient()

                if nil != err {
                        panic(err)
                }

                val, err := client.Get("i1R_reqs_satiSLO_60sec").Result()
                if err != nil {
                        panic(err)
                }
                fmt.Println("wja300: value : ", val)

                instype := []string{"i1", "p2", "p3", "c5"}
                reqtype := []string{"R", "B", "G", "Y", "S"}
                reqtype_detail := []string{"resnet50", "sentiment", "text", "sound", "inception"}

                for i := 0; i < len(reqtype); i++ {
                        for j :=0; j < len(instype); j++{
                                if (sstrings.Contains(api.Name, reqtype_detail[i]) && sstrings.Contains(api.Name, instype[j])){
                                        val, err := client.Get(instype[j]+reqtype[i]+"_scaler").Result()
                                        if err != nil {
                                                panic(err)
                                        }
                                        i, err := strconv.ParseInt(val, 10, 32) // for each req and instype get the number of scaling instances
                                        if err != nil {
                                                panic(err)
                                        }

                                        if(request > int32(i)){ // if inflight cortex decision is larger than scheduler decision
                                                // nothing done
                                        } else { // if scheduler decision is larger than infiight cortex decision
                                                request = int32(i)
                                        }
                                        request = int32(i) // 기존에 왜 주석되어 있는지 기억해야 함....??
                                }
                        }
                        //fmt.Println(i, ":", arr[i])
                }

                fmt.Println("wja300: request :", request)
                fmt.Println("wja300: currentRequestedReplicas :", currentRequestedReplicas)
                fmt.Println("wja300: recommendation :", recommendation)
                fmt.Println("wja300: avgInFlight :", avgInFlight)
                fmt.Println("wja300: rawRecommendation :", rawRecommendation)

		if currentRequestedReplicas != request {
			log.Infof("autoscaling wja30 event: %d -> %d", currentRequestedReplicas, request)
			if err = scaler.Scale(api.Name, request); err != nil {
				return err
			}
		}

		return nil
	}, nil
}
