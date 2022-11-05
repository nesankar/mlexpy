# mlexpy
Simple utilities for handling and managing exploratory and experimental machine learning development.

### Design principles:
1. `mlexpy` is _not_ meant to be a tool deployed in a production prediction environment. Alternatively, it is meant to provide a simple structure to organize different components of machine learning to simplify basic ML exploration and expiramentation, and hopefully improve ML results via standardized and reproducable expiramentation. 

The core goal is to leverage fairly basic, yet powerful, and clear data strucutres and patterns to improve the "workspace" for ML development. Hopefully, this library can tirvialize some common ML development tasks, to allow developers, scientist, and any one else to spend more time in the _investigations_ in ML, and less time in coding or developing a reliable, readable, and reproduceable exploratory codebase / script / notebook.

`mlexpy` provides no explicit ML models or data. Instead it provides various tools to store, interact, and wrap differnet models, methods, or datasets.

#### High level library goals:
- 1. Provide intuitive, standardizeable, and reproduceable ML expiraments.
- 2. Methodological understandability is more important that method simplicity and efficiency. 
    - Becuase this is meant to aid in ML development, often times it is easy to lose track of what explcity steps are and were done in ultimately producing a predicion. `mlexpy` is meant to reduce the amount of code written for ML develpment purely to the explit steps taken in developing a model. For example, Pands DataFrames are currently preffered over numpy nd.arrays simply for column readability.
    - Ideally, `mlexpy` makes is simple to understand exactly what a ML pipeling and model are doing, easing collaboration between engineers, coleuges, and academics.
- 3. `mlexpy` is not developed (yet?) for usage in large scale deep-learning tasks. Perhaps later this will be on interest.

2. `mlexpy` leverages an OOP framework, while this may be less intuitive for some practitioners, the benefits of becoming familiar with some OOP outweigh its learning curve.

Note: Currently, `mlexpy` _only_ provides tooling for supervised learning.


