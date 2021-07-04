# Use of "Generative Adversarial Networks (GANs) to build Intelligent and Interpretable Recognition Systems
## Proposta de Projeto

## Advisor: Hugo Proença

## Objectives

“ Interpretability ” is the key concept in this work proposal. Having interpretable
systems is of maximum importance for many fields, which has been motivating
growing concerns in the research community. Also, the increasingly larger quantities
of data available lead to models of increasingly higher complexity, which responses
are extremely hard to be interpreted by humans. In this context, neural-based methods
are considered a special case of interest, due to this lack of interpretability.

Henceforth, this work proposal aims at designing/developing one solution for
interpretable biometric recognition , which will make easier the application of
biometrics to forensics, by allowing to explain the reasons that sustain a match / non-
match response. This way, the work lays at the intersection of two important scientific
research topics: 1) biometric recognition; and 2) human-machine interaction.


In particular, **the work considers to use Generative Adversarial Frameworks
(GANs)** as the main framework for producing interpretable responses. As biometric
source, **we consider the periocular region** , in order to obtain understandable text
descriptions of: 1) the eyebrows (shape); 2) the eyelids (shape); 3) the iris (color) and
4) the skin (color and texture).

**Figure 2** : **Schematic perspective of a the framework designed for this work
proposal:** Generative Adversarial Network-based solution for developing
interpretable periocular recognition systems. A generator **G** receives human-
understandable descriptions of a biometric sample and is responsible for generating
synthetic samples that accord such descriptions. The discriminator **D** analyzes pairs of
biometric samples and human-understandable descriptions and should discriminate
between the genuine samples with according descriptions and the remaining types of
data (either synthetic samples or genuine samples with wrong descriptions).


