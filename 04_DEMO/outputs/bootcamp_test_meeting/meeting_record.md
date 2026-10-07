# ML Deployment Planning

## Summary
The team reviewed the current ML stack, discussed migrating to PostgreSQL 16, confirmed that the new model will not be deployed this Friday, noted the budget estimate of 2.5 lakh rupees, and assigned Priya to prepare an evaluation report by Monday. Testing remains incomplete and latency results will be reviewed at the next meeting.

## Minutes
- Current ML stack uses PyTorch and PostgreSQL with PGvector
- Proposal to migrate database to PostgreSQL 16
- Decision not to deploy new model this Friday
- Budget estimate clarified as 2.5 lakh rupees
- Testing still pending
- Priya assigned to prepare evaluation report by Monday
- Next meeting will review latency results and GPU configuration

## Decisions
- The new model will not be deployed this Friday — Confirmed | Evidence: We will not deploy the new model this Friday.

## Discussed, Not Decided
- Consider migrating the database to PostgreSQL version 16
- Infrastructure budget estimate is 2.5 lakh rupees
- Review latency results and decide on GPU configuration at next meeting

## Action Items
- Prepare the evaluation report including precision, recall, and F1 score — Owner: Priya; Deadline: Monday; Status: Confirmed | Evidence: Priya will prepare the evaluation report by Monday. The report should include precision, recall, and F1 score.
- Complete testing of the new model — Owner: Unspecified; Deadline: Unspecified; Status: Needs Review | Evidence: We still need to complete testing.
- Review latency results and decide on GPU configuration — Owner: Unspecified; Deadline: Unspecified; Status: Needs Review | Evidence: For the next meeting, we should review the latency results and decide whether the current GPU configuration is sufficient.

## Discussion Points
- Current deployment stack and tools
- Database migration proposal
- Budget clarification

## Open Questions
- Is PostgreSQL version 16 the best choice for the retrieval system?
