# DeepResearch Project Instructions

## Interview Mode

`interview` is the preferred operating mode for this project.

Behavior:

1. At the beginning of a new project or major task, ask the user a short up-front interview to collect the key constraints, goals, and preferences.
2. After that interview is complete, do not keep involving the user in routine execution.
3. Proceed autonomously through local repo work, workspace updates, Tao stage advancement, and normal implementation steps.
4. Only interrupt after the interview when one of these is true:
   - the next step will spend meaningful external money, such as launching paid RunPod compute
   - the next step is destructive or hard to reverse
   - the request is genuinely ambiguous enough that continuing would likely do the wrong thing

Interpretation:

- Treat `interview` as replacing the conversational meaning the user previously referred to as `A0`.
- This is an operating preference for agent behavior, not a Tao config field.

## Paid Compute Discipline

When using paid external compute such as RunPod:

1. Do not leave a paid pod running unless there is a confirmed active setup, training, evaluation, or result-download process.
2. If the controller process is interrupted, fails, or exits, immediately stop the pod unless another confirmed task is still running on it.
3. Do not keep pods alive "just in case" or for debugging convenience.
4. Before ending a turn after any paid-compute work, explicitly verify whether the remote pod is active or idle.
5. If the pod is idle, stop it immediately.
