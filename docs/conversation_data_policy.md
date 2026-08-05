# Conversation data policy

Authenticated chat transcripts are stored so the account owner can revisit a conversation and so
bounded recent context can resolve follow-up questions. Anonymous requests are stateless and are
not written to conversation history.

The default retention period is 90 days after the conversation's latest activity. MongoDB enforces
this with the `conversation_retention_ttl` index on `updatedAt`; operators may shorten the period
with `CONVERSATION_RETENTION_DAYS`. A conversation stores at most 200 messages by default
(`MAX_TRANSCRIPT_MESSAGES`), while only the five most recent turns can enter prompt context.

Only the owning student and administrators can read, rename, pin, or delete a transcript. Missing
ownership metadata and ownership-database failures deny student access. Users can delete individual
conversations through the existing delete endpoint. Logs do not contain question or answer text.
