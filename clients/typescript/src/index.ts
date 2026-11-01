/** Public API of `@agent-adapter/client`. */
export { asVercelTools, toolDefinitions } from "./adapters.ts";
export { ToolClient } from "./client.ts";
export {
  ApprovalAlreadyDecidedError,
  ApprovalDeniedError,
  ApprovalNotFoundError,
  BadRequestError,
  BudgetExceededError,
  EventStreamError,
  ForbiddenError,
  GuardrailBlockedError,
  InternalServerError,
  ModelError,
  NotFoundError,
  RunLimitExceededError,
  ToolArgsInvalidError,
  ToolError,
  ToolFailedError,
  ToolNotFoundError,
  ToolProtocolError,
  ToolServerError,
  ToolTimeoutError,
  ToolTransportError,
  ToolUsageError,
  UnauthorizedError,
} from "./errors.ts";
export type * from "./types.ts";
