# Python signature mode in the Notepad++ panel

Start the prompt with a signature directive, followed by the description:

```text
@python-signature fn_add_13(a, b)
Return the sum of a and b.
```

Click Generate AI Code. The panel and active editor receive the exact header
`def fn_add_13(a, b):` followed by the model-generated body. The final response
used for correction tracking contains that header exactly once. RAG context is
retained. The panel's initial help explains this mode, and its generation status
identifies Python function generation.

The directive is optional. Ordinary prompts follow the existing path. The first
line must contain one function name and plain distinct parameter names; a task
description is required below it. Defaults, annotations, nested expressions,
duplicate parameters and Python keywords are rejected before clearing the prompt
or starting the background worker. Do not use this mode for another language.

The header is supplied by the application, not generated/copied by the model.
It does not ensure a correct function body. The measured experimental checkpoint
matched 16/28 reference ASTs in guided native evaluation; that is a narrow static
regression result, not a guarantee for new tasks.

## Verification

`TestAIPromptPipeline` checks plain prompts, explicit signature normalization,
invalid requests and final header extraction. `TestPluginGenerationFlow` composes
the same prefix preparation and streaming router used by `AIManager` and the
panel, checking editor/panel equality, body backtracking and tracked final output.
Windows CI builds the actual plugin and runs CTest. A real interactive Notepad++
installation check remains a manual check; no GUI recording is claimed.
