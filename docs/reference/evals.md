# Evals

Run eval cases on any agent, grade them and report ([Evaluate your agent](../how-to/evaluate-your-agent.md)).

## `agent_adapter.evals`

::: agent_adapter.evals
    options:
      members: [CASE_EVENT, CallableTarget, CaseChange, CaseResult, ChatTarget, DEFAULT_EVALS_FOLDER, EvalCase, EvalComparison, EvalReport, EvalRun, FlowTarget, GradeResult, Grader, GroupSummary, LLMJudge, Outcome, RecordedRun, RunCase, RunCases, Score, SkippedRun, Target, ToolLoopTarget, as_target, assert_eval, cases_from_runs, eval_executor, graders, load_cases, run_eval, run_eval_sync, save_report, write_cases]

## `agent_adapter.evals.advisor`

::: agent_adapter.evals.advisor
    options:
      members: [AdvisorReport, COMBINED, StepAdvice, StepModels, StepUse, Trial, advise, advise_sync, apply_mapping, current_step, entry_step, step_of]

## `agent_adapter.evals.from_runs`

::: agent_adapter.evals.from_runs
    options:
      members: [ExpectedFrom, case_from_run]

Also exported here, documented under the canonical path: [`RecordedRun`](#agent_adapter.evals.RecordedRun), [`RunCase`](#agent_adapter.evals.RunCase), [`RunCases`](#agent_adapter.evals.RunCases), [`SkippedRun`](#agent_adapter.evals.SkippedRun), [`cases_from_runs`](#agent_adapter.evals.cases_from_runs), [`write_cases`](#agent_adapter.evals.write_cases).

## `agent_adapter.evals.graders`

::: agent_adapter.evals.graders
    options:
      members: [Check, JUDGE_SCHEMA, all_of, code_check, contains, exact, json_schema, llm_judge, max_cost, max_latency, regex, tools_called]

Also exported here, documented under the canonical path: [`Grader`](#agent_adapter.evals.Grader), [`LLMJudge`](#agent_adapter.evals.LLMJudge).
