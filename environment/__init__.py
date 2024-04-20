from gym.envs.registration import register


register(
    id="EchineF/PitsAndOrbs-v0",
    entry_point="environment.pits_and_orbs_env:PitsAndOrbsEnv",
)

register(
    id="EchineF/PitsAndOrbs-two-players-v0",
    entry_point="environment.pits_and_orbs_env:PitsAndOrbsEnv",
    kwargs={"player_num": 2}
)