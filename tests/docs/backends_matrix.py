"""The backends matrix rows: where code runs, how it attaches, and which test proves it.

`tests/docs/test_support_backends.py` renders these rows into docs/support-backends.md.
"""

from __future__ import annotations

from dataclasses import dataclass

NODE = "node"  # a need only a session that checks for Node on PATH meets


@dataclass(frozen=True)
class Check:
    """One test that exercises a backend, and the packages it skips without."""

    test: str  # "tests/...py::test_name"
    needs: tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class Backend:
    """One row: where the code runs, how it attaches, and what proves it."""

    name: str
    attach: str
    extra: str
    guide: str
    examples: tuple[str, ...] = ()
    checks: tuple[Check, ...] = ()
    recipe: str = ""  # why there is no test, when `checks` is empty


BE = "tests/docs/test_backend_examples.py::"
BH = "tests/docs/test_backend_handler_examples.py::"
OTHER = ("examples/backend_other_asgi.py",)
QUEUE = ("examples/backend_queue.py",)
PIPE = ("examples/backend_pipeline.py",)
BOOK = ("examples/backend_notebook.py",)
NB_TEST = Check(f"{BH}test_notebook_cell_streamlit_and_gradio")
PIPE_TEST = Check(
    f"{BH}test_pipeline_task_partition_and_one_backend_per_process", note="no orchestrator"
)

BACKENDS: tuple[Backend, ...] = (
    Backend(
        "FastAPI",
        "`await executor.run(...)` in routes; `create_app` mounts the tool server",
        "`server` (mount only)",
        "backends/fastapi.md",
        ("examples/backend_fastapi.py", "examples/backend_common.py"),
        (
            Check(
                f"{BE}test_fastapi_routes_approve_ask_and_mount_the_tool_server",
                ("fastapi", "httpx", "starlette"),
            ),
        ),
    ),
    Backend(
        "Starlette",
        '`Mount("/agent", app=create_app(registry, ...))`',
        "`server`",
        "backends/other-asgi.md",
        OTHER,
        (Check(f"{BH}test_other_asgi_starlette_mount", ("starlette", "httpx")),),
    ),
    Backend(
        "Litestar, Quart, Sanic",
        "mount `create_app(registry, ...)` as an ASGI app",
        "`server`",
        "backends/other-asgi.md",
        recipe="the same ASGI app is tested under Starlette",
    ),
    Backend(
        "Flask",
        "`run_sync(..., timeout=)` in views; `create_wsgi_app` mounts the tool server",
        "`server-wsgi` (mount only)",
        "backends/flask.md",
        ("examples/backend_flask.py",),
        (
            Check(
                f"{BE}test_flask_views_run_sync_approve_and_mount_the_wsgi_tool_server",
                ("flask", "a2wsgi", "starlette"),
            ),
        ),
    ),
    Backend(
        "Django",
        "`await` in async views, `run_sync` in sync views; `create_app` beside it",
        "`server` (tool server only)",
        "backends/django.md",
        ("examples/backend_django.py",),
        (
            Check(f"{BE}test_django_async_and_sync_views_approve_and_ask", ("django",)),
            Check(
                f"{BE}test_django_asgi_app_serves_django_and_the_tool_server",
                ("django", "starlette", "httpx"),
            ),
        ),
    ),
    Backend(
        "aiohttp",
        "a handler that forwards to `ToolService`",
        "none",
        "backends/other-asgi.md",
        OTHER,
        (
            Check(f"{BH}test_other_asgi_forward_and_grpc_servicer", note="the shim's `forward`"),
            Check(f"{BH}test_other_asgi_aiohttp_shim", ("aiohttp",)),
        ),
    ),
    Backend(
        "gRPC",
        "servicer `await invoke(...)`; sync servers `run_sync` or `LoopRunner`",
        "none",
        "backends/other-asgi.md",
        OTHER,
        (Check(f"{BH}test_other_asgi_forward_and_grpc_servicer", note="fake request, no `grpc`"),),
    ),
    Backend(
        "Celery",
        "`run_sync(..., timeout=)` in the task body",
        "none",
        "backends/celery.md",
        ("examples/backend_celery.py",),
        (
            Check(f"{BE}test_celery_task_bodies_park_and_resume_without_celery"),
            Check(f"{BE}test_celery_tasks_run_eagerly_and_resume_after_review", ("celery",)),
        ),
    ),
    Backend(
        "RQ, Dramatiq, Huey",
        "`LoopRunner().run(lambda: handle_envelope(...))`",
        "none",
        "backends/queues.md",
        QUEUE,
        (Check(f"{BH}test_queue_sync_job_and_activity", note="job body, no broker"),),
    ),
    Backend(
        "Arq, Taskiq",
        "`await handle_envelope(executor, payload, message_id=...)`",
        "none",
        "backends/queues.md",
        QUEUE,
        (
            Check(
                f"{BH}test_queue_consumer_acks_parks_dead_letters_and_redelivers", note="fake queue"
            ),
        ),
    ),
    Backend(
        "Kafka, SQS, Pub/Sub consumers",
        "`await handle_envelope(...)`; ack by reply status",
        "none",
        "backends/queues.md",
        QUEUE,
        (
            Check(
                f"{BH}test_queue_consumer_acks_parks_dead_letters_and_redelivers",
                note="fake queue, no broker",
            ),
            Check(f"{BH}test_queue_pubsub_push_uses_the_stable_message_id"),
        ),
    ),
    Backend(
        "Temporal activities",
        "`await handle_envelope(...)` in the activity",
        "none",
        "backends/queues.md",
        QUEUE,
        (Check(f"{BH}test_queue_sync_job_and_activity", note="activity body, no worker"),),
    ),
    Backend(
        "AWS Lambda",
        "`handler = tool_handler(executor, tools=..., timeout=...)`",
        "none",
        "backends/serverless.md",
        ("examples/backend_serverless.py",),
        (
            Check(
                f"{BH}test_serverless_lambda_parks_reviews_and_resumes_once", note="fake SQS events"
            ),
        ),
    ),
    Backend(
        "Cloud Run functions, Azure Functions, Modal",
        "`await handle_envelope(...)`, or the sync `tool_handler`",
        "none",
        "backends/serverless.md",
        ("examples/backend_serverless.py",),
        (
            Check(
                f"{BH}test_serverless_async_function_returns_problems",
                note="function body, no host runtime",
            ),
        ),
    ),
    Backend(
        "Airflow, Prefect, Dagster",
        "async task `await executor.run_many(...)`; sync `run_sync`",
        "none",
        "backends/pipelines.md",
        PIPE,
        (PIPE_TEST,),
    ),
    Backend(
        "Ray, Spark",
        "`run_sync(...)` in a remote function or a partition body",
        "none",
        "backends/pipelines.md",
        PIPE,
        (PIPE_TEST,),
    ),
    Backend(
        "Jupyter",
        "top-level `await executor.run(...)`, or `run_sync`",
        "none",
        "backends/notebooks.md",
        BOOK,
        (NB_TEST,),
    ),
    Backend(
        "Streamlit",
        "`run_sync(..., timeout=)`; the backend in `st.cache_resource`",
        "none",
        "backends/notebooks.md",
        BOOK,
        (Check(NB_TEST.test, note="fake `st`"),),
    ),
    Backend(
        "Gradio",
        "an async handler that awaits `executor.run(...)`",
        "none",
        "backends/notebooks.md",
        BOOK,
        (Check(NB_TEST.test, note="no `gradio`"),),
    ),
    Backend(
        "Scripts and CLIs",
        "`run_sync(batch)` with `executor.run_many(...)` in `main(argv)`",
        "none",
        "backends/scripts.md",
        ("examples/backend_script.py",),
        (
            Check(f"{BE}test_script_runs_a_batch_denies_unattended_and_sets_the_exit_code"),
            Check(f"{BE}test_script_asks_on_the_terminal_when_interactive"),
        ),
    ),
    Backend(
        "Any warm sync process",
        "`LoopRunner().run(executor.run(...), timeout=...)`",
        "none",
        "concepts/runtime.md",
        ("examples/backend_loop_runner.py",),
        (
            Check(
                "tests/docs/test_deploy_examples.py::test_loop_runner_example_keeps_one_client_warm"
            ),
        ),
    ),
    Backend(
        "Other languages over HTTP",
        "`create_app(registry, ...)`; REST + SSE from any client",
        "`server`",
        "cross-language/http-server.md",
        ("examples/serve_tools.py",),
        (
            Check("tests/server/test_examples_server.py::test_serve_tools_demo"),
            Check(
                "tests/server/test_examples_server.py::test_serve_tools_builds_the_app",
                ("starlette", "mcp"),
            ),
        ),
    ),
    Backend(
        "Other languages over MCP",
        "`streamable_http_app(registry)` at `/mcp`",
        "`mcp`",
        "cross-language/mcp.md",
        ("examples/mcp_http.py",),
        (
            Check(
                "tests/integrations/mcp/test_examples.py::test_mcp_http_builds_the_app_at_mcp",
                ("mcp",),
            ),
        ),
    ),
    Backend(
        "TypeScript client",
        "`@agent-adapter/client` against `create_app`",
        "`server`",
        "cross-language/typescript-client.md",
        ("clients/typescript/examples/quickstart.ts",),
        (
            Check(
                "tests/server/test_ts_client.py::test_typescript_client_against_the_live_server",
                ("starlette", "uvicorn", NODE),
                note="runs `node --test`",
            ),
        ),
    ),
)
