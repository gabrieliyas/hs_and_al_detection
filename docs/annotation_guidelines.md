# Annotation Guidelines — Custom Contextual Dataset

## 1. Label definitions (N / AL / HS)

Read the target post together with as much of the available parent-message
context as you need (see Section 2 for when to use context). Assign exactly
one label.

**Neither (N)**
The post does not insult, demean, attack, or express hatred toward any
person or group, even once context is considered. Ordinary disagreement,
criticism of ideas, policies, or public figures without personal insult,
and neutral commentary all fall here.

**Abusive (AL)**
The post is insulting, demeaning, vulgar, or hostile toward a specific
person or group, but the hostility is NOT based on the target's
membership in a protected group (see HS below). This includes general
insults, profanity directed at a person, and hostility tied to something
other than protected-attribute identity (e.g., mocking someone's opinion,
intelligence, or behavior in a demeaning way).

**Hate Speech (HS)**
The post attacks, demeans, or expresses hatred toward a person or group
specifically on the basis of a protected attribute — religion, ethnicity/
race, gender or sexual orientation, disability/physical condition, or a
comparable protected-group characteristic — whether the attack is
explicit or only apparent once context is taken into account (e.g.,
coded language, escalation, or a resolved ambiguous reference).

**Priority rule:** if a post meets the criteria for HS, code it as HS
regardless of whether it also contains abusive-language markers
(vulgarity, general insults). This follows the convention that hate
speech is treated as the narrower, more severe category that takes
precedence when both apply (cf. Davidson et al., 2017).

## 2. When is context needed? (context_needed = 0 / 1)

Set **context_needed = 1** if you cannot confidently assign a label using
the target post text alone, and had to read one or more parent messages
to resolve any of the following:

- **Ambiguous reference:** who or what "orang seperti kamu," "dia," "itu,"
  etc. actually refers to.
- **Sarcasm/irony:** the literal wording contradicts the apparent
  intended meaning.
- **Coded language/euphemism:** a surface-neutral phrase carries a
  harmful meaning known from the preceding conversation.
- **Escalation:** the post's harmfulness only becomes apparent as part of
  an escalating pattern across the thread.
- **Opinion-attack ambiguity:** whether the post is a general opinion
  statement or a targeted personal attack.
- **Quotation/stance:** whether the post endorses or opposes a quoted or
  reported statement from an earlier message.

Set **context_needed = 0** if the target post alone is sufficient to
confidently assign the label — including posts that are explicitly and
directly worded (no ambiguity to resolve).

## 3. Tie-break / adjudication protocol

**For the 3-class label (N/AL/HS):**
- All 3 annotators agree -> status = *Bulat* (unanimous). Use this label.
- 2 of 3 agree -> status = *Mayoritas* (majority). Use the majority label.
- All 3 give different labels (possible only with exactly 3 classes and 3
  annotators) -> status = *Sengketa* (disputed). Escalate to a designated
  4th adjudicator (e.g., thesis advisor or a separate reviewer), who reads
  the target post and available context and gives the final ruling. Record
  the adjudicator's label as final; keep the status flagged as adjudicated
  for reporting in Chapter III.

**For context_needed (binary):**
- All 3 agree -> *Bulat*.
- 2 of 3 agree -> *Mayoritas*, use majority value.
- (A 3-way split is not possible for a binary variable with 3 annotators,
  so no adjudication step is needed here.)

Report the label-agreement status and the context-needed-agreement status
as two separate columns — they will often diverge.

## 4. Target group (descriptive field, single-annotator)

Use the following generalized scheme (applies across all 3 labels, not
just HS, since Abusive-language posts can target someone without
targeting a protected attribute):

- `individual` — targets a specific named or clearly identifiable person,
  not on the basis of protected-group membership.
- `protected_group` — targets a group defined by a protected attribute
  (religion, ethnicity/race, gender, disability, etc.). Optionally note
  which attribute in a free-text comment for descriptive richness — this
  does not need its own structured column.
- `non_protected_group` — targets a group not defined by a protected
  attribute (e.g., a political party, a fanbase, a profession).
- `institution_policy_idea` — targets an institution, policy, or idea
  rather than a person or group.
- `none` — no identifiable target (typically Neither-labeled posts).
- `unclear` — a target exists but cannot be confidently categorized.

This field does not require a separate Fleiss' kappa — it is descriptive
context for the qualitative analysis in Chapter IV, not a modelling
target.
