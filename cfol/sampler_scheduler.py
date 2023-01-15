from cfol.class_sampler import ClassSampler


class SamplerResetScheduler(object):
    def __init__(self, sampler: ClassSampler, milestones) -> None:
        self._step_count = 0
        self.milestones = milestones.copy()
        self.sampler = sampler

    def step(self):
        self._step_count +=1

        if len(self.milestones) > 0:
            next_milestone = self.milestones[0]
            if self._step_count > next_milestone:
                print("Resetting ClassSampler prior")
                self.sampler.reset_prior()
                self.milestones = self.milestones[1:]
