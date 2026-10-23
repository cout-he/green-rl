"""Forecast-only supply bias for the bounded, frozen-policy H1 pilot.

The physical object and its greenPowerList remain untouched. An estimate's
target hour decides whether it is in the future, even when the author's
repeating window maps that hour back to the observed current source hour.
"""

import math


class ForecastPowerProxy:
    """Expose biased future forecasts, retaining the original physical methods.

    ``bias`` must be finite and greater than -1, so its multiplier is positive.
    The H1 runner restricts this general interface to 0 and +0.20. Counters count
    a biased segment only when its duration and source supply are positive and
    a nonzero bias changes that supply; a zero-duration endpoint is not counted.
    """

    def __init__(self, physical, bias, current_time):
        bias = float(bias)
        if not math.isfinite(bias) or bias <= -1:
            raise ValueError("bias must be finite and greater than -1")
        self.physical = physical
        self.bias = bias
        self._current_time = current_time
        self._counters = {"slot_calls": 0, "estimate_calls": 0,
                          "biased_slot_segments": 0, "biased_estimate_segments": 0}

    @property
    def metadata(self):
        return {"bias": self.bias, "future_multiplier": 1 + self.bias,
                "bias_scope": "future_target_hour",
                "physical_supply_mutated": False}

    @property
    def counters(self):
        result = self._counters.copy()
        result["biased_future_segments"] = (result["biased_slot_segments"]
                                             + result["biased_estimate_segments"])
        return result

    def __getattr__(self, name):
        return getattr(self.physical, name)

    def getGreenPowerSlot(self, currentTime):
        self._counters["slot_calls"] += 1
        slots = self.physical.getGreenPowerSlot(currentTime)
        if self.bias == 0:
            return slots
        result = []
        for offset, slot in enumerate(slots):
            predicted = slot.copy()
            if offset > 0:
                predicted["power"] = slot["power"] * (1 + self.bias)
                if (slot["lastTime"] > 0 and slot["power"] > 0
                        and predicted["power"] != slot["power"]):
                    self._counters["biased_slot_segments"] += 1
            result.append(predicted)
        return result

    def getGreenUtiliEstimate(self, power, start, end, minIndex, maxIndex):
        self._counters["estimate_calls"] += 1
        if self.bias == 0:
            return self.physical.getGreenUtiliEstimate(power, start, end, minIndex, maxIndex)
        current_index = int(self._current_time() / 3600)
        usedGreen = 0
        startIndex = int(start / 3600)
        endIndex = int(end / 3600)
        t = start
        # Preserve the author's inclusive endpoint, segment order, and wrapping.
        for i in range(startIndex, endIndex + 1):
            if i == endIndex:
                lastTime = end - t
            else:
                lastTime = (i + 1) * 3600 - t
            if i > maxIndex:
                greenIndex = minIndex + (i - minIndex) % (maxIndex - minIndex + 1)
            else:
                greenIndex = i
            supply = self.physical.greenPowerList[greenIndex]
            if i > current_index:
                predicted_supply = supply * (1 + self.bias)
                if lastTime > 0 and supply > 0 and predicted_supply != supply:
                    self._counters["biased_estimate_segments"] += 1
                supply = predicted_supply
            if power >= supply:
                usedGreen += supply * lastTime
            else:
                usedGreen += power * lastTime
            t = (i + 1) * 3600
        return usedGreen

    def getGreenUtili(self, power, start, end):
        return self.physical.getGreenUtili(power, start, end)

    def getGreenPowerUtilization(self, powerSlot):
        # Binding to physical keeps its internal self.getGreenUtili physical too.
        return self.physical.getGreenPowerUtilization(powerSlot)


def install_forecast(env, bias):
    """Install a forecast view whose clock follows the environment's live time."""
    if isinstance(env.cluster.greenPower, ForecastPowerProxy):
        raise ValueError("a forecast proxy is already installed")
    proxy = ForecastPowerProxy(env.cluster.greenPower, bias, lambda: env.current_timestamp)
    env.cluster.greenPower = proxy
    return proxy
