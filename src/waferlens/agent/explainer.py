"""L5 explanation agent (interface).

Given a classification result, the L3 statistics behind it and the L4 SPC
state, produce a short explanation: what pattern, which candidate process
causes (from the README association table, framed as hypotheses), and what to
check next. All numbers pass waferlens.agent.factcheck before output.

TODO(owner):
  1. Prompt assembly from (predicted label, per-class probabilities, SPC
     state) — causes stated as hypotheses to verify, never as conclusions.
  2. Claim extraction: parse the numbers out of the draft explanation and run
     factcheck.check_numbers against the computed stats; on mismatch, retry
     once with the corrections, then refuse with the mismatch list attached.
  3. Ten fixed evaluation cases, human-scored for process plausibility;
     scores committed under results/.
"""
