# ATT supplementary experiment code

| Experiment | Entry points |
|---|---|
| Exp1 · Common judge across five views | [Prepare inputs](Exp1/prepare_common_judge.py), [run or score](Exp1/run_common_judge.py), [Windows launcher](Exp1/start_common_judge.ps1) |
| Exp2 · Stale observation recovery | [Scripted reachability](Exp2/run_stale_reachability.py), [paired model runs](Exp2/run_stale_models.py), [Windows launcher](Exp2/start_stale_models.ps1) |
| Exp3 · Indistinguishable traces and log selection | [Enumerate and evaluate](Exp3/run_log_selection.py) |
| Exp4 · Equal decision budgets and Oracle intervention | [Prepare design](Exp4/prepare_design.py), [run experiment](Exp4/run_experiment.py), [Windows launcher](Exp4/start_experiment.ps1) |

[Prepare frozen audit inputs](prep_e0.py) derives Exp1 inputs from locally supplied historical trajectories. Exp2 uses the matching local Recovery case file. Exp3 requires a local `Exp3/design.json`. Exp4 uses locally prepared task manifests and current sandbox and local MCP qualification. The scripts expect these inputs in the paths defined by their entry points.

Supply model credentials through the process environment. Input datasets, generated manifests, traces, databases, reports, and run logs are local artifacts and are excluded by the repository ignore rules.
