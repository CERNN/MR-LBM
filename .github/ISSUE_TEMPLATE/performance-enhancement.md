---
name: Performance Enhancement Request
about: Propose ideas to improve the performance of the codebase
title: "[Performance] "
labels: enhancement, performance
assignees: ''

---

## Performance Enhancement Request

**What part of the code could be optimized?**  
Describe the section, module, or process you believe could be made more efficient.

> Example: "The collision step in the LBM kernel seems to dominate runtime in large domains."

---

**Why do you think this is a performance bottleneck?**  
Explain your reasoning—this could be based on intuition, experience, or a general understanding of the current implementation.

> Example: "It performs many redundant memory accesses and has no parallelization."

---

**Do you have suggestions for improving it?**  
Describe any techniques or ideas that could be explored to improve performance (even if you're unsure how to implement them).

> Example: "Investigate OpenMP or SIMD usage. Could also look into memory layout optimization."

---

**Are you planning to contribute to this enhancement?**  
Indicate your level of involvement with this request.

- [ ] I plan to work on this
- [ ] I’m looking for collaborators
- [ ] I’m only opening this as a suggestion

---

**Additional context**  
Add any links, references, or notes that help explain or justify the request.

---
