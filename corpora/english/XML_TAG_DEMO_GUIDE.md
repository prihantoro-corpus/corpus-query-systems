# XML Tag Demo Corpus - Testing Guide

This corpus demonstrates XML tag-based search with rich inline markup.

## Available Tags

### Named Entities
- `<PN>` - Person/Place names
  - Attributes: `type` (person/place), `role`, `country`, `sport`, `species`
- `<ORG>` - Organizations
  - Attributes: `type` (company/bank/team/university/journal), `sector`, `sport`

### Numbers & Time
- `<NUM>` - Numerical values
  - Attributes: `type` (currency/quantity/score/percentage), `unit`
- `<TIME>` - Temporal expressions
  - Attributes: `type` (year)

### Semantic Markup
- `<EVAL>` - Evaluative language
  - Attributes: `sentiment` (positive/negative), `intensity`, `certainty`
- `<TERM>` - Technical terms
  - Attributes: `type` (protein/organism)
- `<TECH>` - Technology/devices
  - Attributes: `type` (vehicle/device), `category`
- `<PROCESS>` - Processes
  - Attributes: `type`
- `<ENTITY>` - Other entities
  - Attributes: `type`, `threat`

## Example Queries

### Basic Tag Queries
1. `<PN>` → All person and place names
2. `<ORG>` → All organizations
3. `<NUM>` → All numbers
4. `<EVAL>` → All evaluative words

### Filtered by Attributes
5. `<PN type="person">` → Only people (Sarah Johnson, Michael Chen, etc.)
6. `<PN type="place">` → Only places (Silicon Valley, Andromeda Nebula)
7. `<ORG type="company">` → Only companies (TechCorp Inc)
8. `<ORG type="team">` → Only sports teams
9. `<NUM type="currency">` → Only monetary values
10. `<EVAL sentiment="positive">` → Only positive evaluations

### Multiple Attributes
11. `<PN type="person" role="CEO">` → CEOs only (Sarah Johnson)
12. `<PN type="person" role="athlete">` → Athletes (David Martinez)
13. `<ORG type="team" sport="football">` → Football teams
14. `<EVAL sentiment="positive" intensity="high">` → Highly positive evaluations

### Wildcards
15. `<PN role="*er">` → Roles ending in "er" (researcher)
16. `<ORG type="*">` → All organizations (matches any type)
17. `<NUM type="qu*">` → quantity numbers

### Multiword Combinations
18. `at <ORG type="university">` → "at MIT"
19. `<PN type="person"> scored` → "David Martinez scored"
20. `the <TECH>` → "the Starfire", "the quantum scanner"
21. `<NUM> new jobs` → "200 new jobs"

### Complex Queries
22. `<PN type="person" role="CEO"> announced` → CEO announcements
23. `<EVAL sentiment="positive"> performance` → positive performance mentions
24. `<ORG type="team"> ended` → team match results

## Testing in Cortex

1. **Import the corpus**: Use "Upload Corpus" → select `xml_tag_demo.xml`
2. **Go to Concordance**: Try the example queries above
3. **Verify results**: Check that only matching tokens are returned
4. **Try combinations**: Mix XML tags with other syntax like wildcards, POS tags, etc.
