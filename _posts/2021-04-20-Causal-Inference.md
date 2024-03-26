---
layout: post
title:  "Introduction to Causal Inference"
date:   2021-04-20 08:43:59
author: Rittik Ghosh
categories: Causal
tags:	jekyll welcome
cover:  "/assets/instacode.png"
---

# DoWhy

<img src="causal_assets/cnc_2.png">

<img src="causal_assets/cnc_1.png">

### Correlation does not equal causation

<img src="causal_assets/cnc_3.png">

### Kind of obvious?

The human mind has a remarkable ability to associate causes with a specific event. From the outcome of an election to an object dropping on the floor, we are constantly associating chains of events that cause a specific effect. <b>Neuropsychology refers to this cognitive ability as causal reasoning.</b>

### Causal Inference vs typical Machine Learning?

Machine Learning (ML)-based projects focus on predicting outcomes rather than understanding causality. <b>They are outstanding correlation machines.</b>

<img src="causal_assets/mlnow.png">

However deep your neural network is, most of the patterns it’s matching are likely devoid of true understanding of the latent factors resulting in the <b> “why” </b> of what the data.

<b>Uploaded new Logo >> Downloads increased by 2X</b>
Did downloads increase because of the new images in your app stores? Or did they just happen to occur at the same time?

<img src="causal_assets/ml_end_1.png">


<img src="causal_assets/ml_end_2.png">

### Causal Inference today

The challenge with causal inference is not that is a new discipline, quite the opposite, but that the current methods represent a very small and simplistic version of causal reasoning. Despite that frequent usage in social-economics, medical research, and other social sciences.

<img src="causal_assets/end.png">

## dowhy

### What is a causal effect?

#### Key Ingredient: counterfactual reasoning

Suppose that we want to find the causal effect of taking an action A on the outcome Y. To define the causal effect, consider two worlds: 1. World 1 (Real World): Where the action A was taken and Y observed 2. World 2 (Counterfactual World): Where the action A was not taken (but everything else is the same)

Causal effect is the difference between Y values attained in the real world versus the counterfactual world.



<img src="causal_assets/one.png">

In other words, A causes Y iff changing A leads to a change in Y, <b>keeping everything else constant.</b>

<img src="causal_assets/two.png">

For example, a new marketing campaign may be deployed during the holiday season, a new feature may only have been applied to high-activity users, or the older patients may have been more likely to receive the new drug, and so on. <b>The goal of causal inference methods is to remove such correlations and confounding from the data and estimate the true effect of an action</b>, as given by the equation above.



### Methodology

<img src="causal_assets/dowhy.png">

<ul>
<li>
Model: DoWhy models each problem using a graph of causal relationships. The graph might include prior knowledge of the causal relationships in the variables but DoWhy does not make any immediate assumptions.
</li> 
<li>    
Identify: Using the input graph, DoWhy finds all possible ways of identifying a desired causal effect based on the graphical model. It uses graph-based criteria and do-calculus to find potential ways find expressions that can identify the causal effect
</li>
<li>    
Estimate: DoWhy estimates the causal effect using statistical methods such as matching or instrumental variables. The current version of DoWhy supports estimation methods based such as propensity-based-stratification or propensity-score-matching that focus on estimating the treatment assignment as well as regression techniques that focus on estimating the response surface.
</li>  
<li>    
Verify: Finally, DoWhy uses different robustness methods to verify the validity of the causal effect.
</li>    
</ul>    


```python
import numpy as np
import pandas as pd

import dowhy
from dowhy import CausalModel
import dowhy.datasets

# Avoid printing dataconversion warnings from sklearn
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# Config dict to set the logging level
import logging.config
DEFAULT_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'loggers': {
        '': {
            'level': 'WARN',
        },
    }
}

logging.config.dictConfig(DEFAULT_LOGGING)
```


```python
data = dowhy.datasets.linear_dataset(beta=10,
        num_common_causes=5,
        num_instruments = 2,
        num_effect_modifiers=0,
        num_samples=20000,
        treatment_is_binary=True,
        num_discrete_common_causes=1)
df = data["df"]
print(df.head())
print(data["dot_graph"])
print("\n")
print(data["gml_graph"])
```

        Z0        Z1        W0        W1        W2        W3 W4    v0          y
    0  0.0  0.113322 -0.569917 -0.850208  0.780494 -2.341097  2  True  13.661543
    1  1.0  0.162743  0.536346  0.536491 -0.199332 -2.037995  0  True  13.336372
    2  0.0  0.471451  0.709805 -1.295933 -1.392377 -2.399189  2  True  14.629494
    3  1.0  0.510568 -0.831252 -0.065517 -0.556490 -0.775688  3  True  16.245827
    4  0.0  0.607321  0.355726  0.735634  1.094497 -2.076653  0  True  15.010771
    digraph { U[label="Unobserved Confounders"]; U->y;v0->y;U->v0;W0-> v0; W1-> v0; W2-> v0; W3-> v0; W4-> v0;Z0-> v0; Z1-> v0;W0-> y; W1-> y; W2-> y; W3-> y; W4-> y;}
    
    
    graph[directed 1node[ id "y" label "y"]node[ id "Unobserved Confounders" label "Unobserved Confounders"]edge[source "Unobserved Confounders" target "y"]node[ id "W0" label "W0"] node[ id "W1" label "W1"] node[ id "W2" label "W2"] node[ id "W3" label "W3"] node[ id "W4" label "W4"]node[ id "Z0" label "Z0"] node[ id "Z1" label "Z1"]node[ id "v0" label "v0"]edge[source "Unobserved Confounders" target "v0"]edge[source "v0" target "y"]edge[ source "W0" target "v0"] edge[ source "W1" target "v0"] edge[ source "W2" target "v0"] edge[ source "W3" target "v0"] edge[ source "W4" target "v0"]edge[ source "Z0" target "v0"] edge[ source "Z1" target "v0"]edge[ source "W0" target "y"] edge[ source "W1" target "y"] edge[ source "W2" target "y"] edge[ source "W3" target "y"] edge[ source "W4" target "y"]]


### Modeling:


```python
model=CausalModel(
        data = df,
        treatment=data["treatment_name"],
        outcome=data["outcome_name"],
        graph=data["gml_graph"]
        )

```

Confounders: These are variables that cause both the action and the outcome. As a result, any observed correlation between the action and the outcome may simply be due to the confounder variables, and not due to any causal relationship from the action to the outcome.

Instrumental Variables: These are special variables that cause the action, but do not directly affect the outcome. In addition, they are not affected by any variable that affects the outcome. Instrumental variables can help reduce bias, if used in the correct way.


```python
model.view_model()
```


    
![png](/assets/causal_assets/output_32_0.png)
    


### Identifying:


```python
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimand)
```

    Estimand type: nonparametric-ate
    
    ### Estimand : 1
    Estimand name: backdoor
    Estimand expression:
      d                                 
    ─────(Expectation(y|W4,W0,W3,W2,W1))
    d[v₀]                               
    Estimand assumption 1, Unconfoundedness: If U→{v0} and U→y then P(y|v0,W4,W0,W3,W2,W1,U) = P(y|v0,W4,W0,W3,W2,W1)
    
    ### Estimand : 2
    Estimand name: iv
    Estimand expression:
    Expectation(Derivative(y, [Z0, Z1])*Derivative([v0], [Z0, Z1])**(-1))
    Estimand assumption 1, As-if-random: If U→→y then ¬(U →→{Z0,Z1})
    Estimand assumption 2, Exclusion: If we remove {Z0,Z1}→{v0}, then ¬({Z0,Z1}→y)
    
    ### Estimand : 3
    Estimand name: frontdoor
    No such variable found!
    


### Estimating:


```python
causal_estimate = model.estimate_effect(identified_estimand,
        method_name="backdoor.propensity_score_stratification")
print(causal_estimate)
print("Causal Estimate is " + str(causal_estimate.value))
```

    *** Causal Estimate ***
    
    ## Identified estimand
    Estimand type: nonparametric-ate
    
    ### Estimand : 1
    Estimand name: backdoor
    Estimand expression:
      d                                 
    ─────(Expectation(y|W4,W0,W3,W2,W1))
    d[v₀]                               
    Estimand assumption 1, Unconfoundedness: If U→{v0} and U→y then P(y|v0,W4,W0,W3,W2,W1,U) = P(y|v0,W4,W0,W3,W2,W1)
    
    ## Realized estimand
    b: y~v0+W4+W0+W3+W2+W1
    Target units: ate
    
    ## Estimate
    Mean value: 9.971438684580926
    
    Causal Estimate is 9.971438684580926


### Refuting:

#### Adding a random common cause variable:


```python
res_random=model.refute_estimate(identified_estimand, causal_estimate, method_name="random_common_cause")
print(res_random)
```

    Refute: Add a Random Common Cause
    Estimated effect:9.971438684580926
    New effect:9.969951001340512
    


#### Replacing treatment with a random (placebo) variable


```python
res_placebo=model.refute_estimate(identified_estimand, causal_estimate,
        method_name="placebo_treatment_refuter", placebo_type="permute")
print(res_placebo)
```

    Refute: Use a Placebo Treatment
    Estimated effect:9.971438684580926
    New effect:0.007638599300858714
    p value:0.48
    


#### Removing a random subset of the data


```python
res_subset=model.refute_estimate(identified_estimand, causal_estimate,
        method_name="data_subset_refuter", subset_fraction=0.9)
print(res_subset)

```

    Refute: Use a subset of data
    Estimated effect:9.971438684580926
    New effect:9.934541775672317
    p value:0.24
    


#### Causal question: What is the impact of offering the membership rewards program on total sales?

#### Counterfactual question: If the current members had not signed up for the program, how much less would they have spent on the website?


```python
i=3
```


```python
num_users = 100000
num_months = 12

signup_months = np.random.choice(np.arange(1, num_months), num_users) * np.random.randint(0,2, size=num_users)
df = pd.DataFrame({
    'user_id': np.repeat(np.arange(num_users), num_months),
    'signup_month': np.repeat(signup_months, num_months), # signup month == 0 means customer did not sign up
    'month': np.tile(np.arange(1, num_months+1), num_users), # months are from 1 to 12
    'spend': np.random.poisson(500, num_users*num_months) #np.random.beta(a=2, b=5, size=num_users * num_months)*1000 # centered at 500
})
# Assigning a treatment value based on the signup month 
df["treatment"] = (1-(df["signup_month"]==0)).astype(bool)
# Simulating effect of month (monotonically increasing--customers buy the most in December)
df["spend"] = df["spend"] - df["month"]*10
# The treatment effect (simulating a simple treatment effect of 100)
after_signup = (df["signup_month"] < df["month"]) & (df["signup_month"] !=0)
df.loc[after_signup,"spend"] = df[after_signup]["spend"] + 100
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>signup_month</th>
      <th>month</th>
      <th>spend</th>
      <th>treatment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>7</td>
      <td>1</td>
      <td>478</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>7</td>
      <td>2</td>
      <td>509</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>475</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>7</td>
      <td>4</td>
      <td>473</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>7</td>
      <td>5</td>
      <td>434</td>
      <td>True</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1199995</th>
      <td>99999</td>
      <td>1</td>
      <td>8</td>
      <td>498</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1199996</th>
      <td>99999</td>
      <td>1</td>
      <td>9</td>
      <td>517</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1199997</th>
      <td>99999</td>
      <td>1</td>
      <td>10</td>
      <td>504</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1199998</th>
      <td>99999</td>
      <td>1</td>
      <td>11</td>
      <td>481</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1199999</th>
      <td>99999</td>
      <td>1</td>
      <td>12</td>
      <td>507</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>1200000 rows × 5 columns</p>
</div>




```python
causal_graph = """digraph {
treatment[label="Program Signup in month i"];
pre_spends;
post_spends;
Z->treatment;
U[label="Unobserved Confounders"]; 
pre_spends -> treatment;
treatment->post_spends;
signup_month->post_spends; signup_month->pre_spends;
signup_month->treatment;
U->treatment; U->pre_spends; U->post_spends;
}"""

# Post-process the data based on the graph and the month of the treatment (signup)
df_i_signupmonth = df[df.signup_month.isin([0,i])].groupby(["user_id", "signup_month", "treatment"]).apply(
    lambda x: pd.Series({'pre_spends': np.sum(np.where(x.month < i, x.spend,0))/np.sum(np.where(x.month<i, 1,0)),
                        'post_spends': np.sum(np.where(x.month > i, x.spend,0))/np.sum(np.where(x.month>i, 1,0)) })
).reset_index()
print(df_i_signupmonth)
model = dowhy.CausalModel(data=df_i_signupmonth,
                     graph=causal_graph.replace("\n", " "),
                     treatment="treatment",
                     outcome="post_spends")
```

    2021-03-09 12:43:58,717 [16688] ERROR    dowhy.causal_graph:61: [JupyterRequire] Error: Pygraphviz cannot be loaded. No module named 'pygraphviz'
    Trying pydot ...


           user_id  signup_month  treatment  pre_spends  post_spends
    0            1             0      False       473.5   410.444444
    1            2             0      False       482.0   421.111111
    2            4             0      False       494.5   418.111111
    3            5             0      False       486.0   408.555556
    4            6             3       True       477.5   521.555556
    ...        ...           ...        ...         ...          ...
    54658    99985             0      False       506.5   414.111111
    54659    99989             0      False       490.0   430.000000
    54660    99995             0      False       483.0   419.888889
    54661    99996             0      False       494.0   425.888889
    54662    99998             0      False       498.0   422.888889
    
    [54663 rows x 5 columns]



```python
model.view_model()
```

    2021-03-09 12:43:58,748 [16688] WARNING  dowhy.causal_graph:87: [JupyterRequire] Warning: Pygraphviz cannot be loaded. Check that graphviz and pygraphviz are installed.



    
![png](/assets/causal_assets/output_49_1.png)
    



```python
"Percent of users signed up in month {}: {}%".format(i,np.round(100*(df_i_signupmonth.treatment.sum()/len(df_i_signupmonth))),4)
```




    'Percent of users signed up in month 3: 8.0%'




```python
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimand)
```

    2021-03-09 12:43:59,052 [16688] WARNING  dowhy.causal_identifier:313: [JupyterRequire] If this is observed data (not from a randomized experiment), there might always be missing confounders. Causal effect cannot be identified perfectly.


    Estimand type: nonparametric-ate
    
    ### Estimand : 1
    Estimand name: backdoor
    Estimand expression:
         d                                                        
    ────────────(Expectation(post_spends|pre_spends,signup_month))
    d[treatment]                                                  
    Estimand assumption 1, Unconfoundedness: If U→{treatment} and U→post_spends then P(post_spends|treatment,pre_spends,signup_month,U) = P(post_spends|treatment,pre_spends,signup_month)
    
    ### Estimand : 2
    Estimand name: iv
    Estimand expression:
    Expectation(Derivative(post_spends, [Z])*Derivative([treatment], [Z])**(-1))
    Estimand assumption 1, As-if-random: If U→→post_spends then ¬(U →→{Z})
    Estimand assumption 2, Exclusion: If we remove {Z}→{treatment}, then ¬({Z}→post_spends)
    
    ### Estimand : 3
    Estimand name: frontdoor
    No such variable found!
    



```python
estimate = model.estimate_effect(identified_estimand, 
                                 method_name="backdoor1.propensity_score_matching",
                                target_units="att")
print(estimate)
```

    *** Causal Estimate ***
    
    ## Identified estimand
    Estimand type: nonparametric-ate
    
    ## Realized estimand
    b: post_spends~treatment+pre_spends+signup_month
    Target units: att
    
    ## Estimate
    Mean value: 98.33248050682258
    



```python
refutation = model.refute_estimate(identified_estimand, estimate, method_name="placebo_treatment_refuter",
                     placebo_type="permute", num_simulations=20)
print(refutation)
```

    2021-03-09 12:56:05,833 [16688] WARNING  dowhy.causal_refuters.placebo_treatment_refuter:142: [JupyterRequire] We assume a Normal Distribution as the sample has less than 100 examples.
                     Note: The underlying distribution may not be Normal. We assume that it approaches normal with the increase in sample size.


    Refute: Use a Placebo Treatment
    Estimated effect:98.33248050682258
    New effect:-0.21094907407407465
    p value:0.3459052885854157
    


<img src="causal_assets/ending.png">
