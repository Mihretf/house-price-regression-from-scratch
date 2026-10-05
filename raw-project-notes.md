Step 1: Choosing a dataset



We are going to choose a dataset from kaggle. We want a dataset that has a lot of rows meaning a lot of houses with features that are affecting its price. There are around seven features. 

After choosing the dataset on kaggle the next step would be to inspect the dataset. The reason we would inspect a dataset is to check if there are are missing values in our data. Any duplicate files, the column names, the data types this particular dataset is using and any other relations or features or distributions that we learn from it. 

We can download our file and write a small python script to check these things locally, or we can use kaggle or jupyter notebooks. In this particular project, to show hands on involvement we will be working locally on our computer. 

Steps: 
Download the kaggle dataset by CSV version, by downloading dataset as zip file. 
Create and open a dedicated project folder and extract the zip file there. 
Open the folder inside VScode or the code editor you are using, then create a new folder called scripts, where you will add all the python scripts we will use in this project. It is important to keep being organized. Keep the data inside a folder called dataset as well. 
Inside the scripts folder create an inspection.py file. 
Inside our inspection file we are checking for the following values
A. Checking the shape, meaning our number of rows and columns
B. listing the columns we have
C. Listing the type of datas we have
D. Checking for missing and duplicate values
E. Looking at the basic statistics of our data like the mean, standard deviation, the min and max and percentiles for numerical columns. 

Results gained:
 Barney@MERCY MINGW64 ~/AI-projects/project-1-supervised-learning
$ python scripts/inspection.py
Dataset shape:
(1000, 8)

Column names:
['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 'Year_Built', 'Lot_Size', 'Garage_Size', 'Neighborhood_Quality', 'House_Price']

Data types:
Square_Footage            int64
Num_Bedrooms              int64
Num_Bathrooms             int64
Year_Built                int64
Lot_Size                float64
Garage_Size               int64
Neighborhood_Quality      int64
House_Price             float64
dtype: object

Missing values:
Square_Footage          0
Num_Bedrooms            0
Num_Bathrooms           0
Year_Built              0
Lot_Size                0
Garage_Size             0
Neighborhood_Quality    0
House_Price             0
dtype: int64

Number of duplicate rows:
0

Basic statistics:
       Square_Footage  ...   House_Price
count     1000.000000  ...  1.000000e+03
mean      2815.422000  ...  6.188610e+05
std       1255.514921  ...  2.535681e+05
min        503.000000  ...  1.116269e+05
25%       1749.500000  ...  4.016482e+05
50%       2862.500000  ...  6.282673e+05
75%       3849.500000  ...  8.271413e+05
max       4999.000000  ...  1.108237e+06

[8 rows x 8 columns]
(.venv) 
Barney@MERCY MINGW64 ~/AI-projects/project-1-supervised-learning
$ 


—
So our initial inspection showed that the dataset is already relatively clean. Therefore, we do not need to perform missing-value imputation or duplicate removal at this stage. 

Now instead of just reading just the summary pages, we need to see the data as well. So we will use matplot lib that will plot the distribution of each numerical column. 

This is important because it lets us see the distribution of a variable, say for example a house size, based on there square kilometer we can see if most houses are smaller or large? Observation that we need to see visually we can have them here. We created an output folder so that the image or the png of the data distributions can be seen there. 

Then we are also going to develop a step to see the relationships between our feature and the target variable which is the house price. So, we’re seeing seeing how each feature relate to house price. 

The reason we do this is because first off we are trying to understand what our model will have to learn before actually training the model. We are going to see this observations in our plots and later we can be able to see if our model learn the relationships we observed in the data. 

And eventually it will help us question and get answer to a lot of things as well. Like how well can asimple linear model capture the relationship in this dataset, If the scatterplot is a linear relation, then linear regression might make sense but if we see curved relationships then we might need to look more into polynomial regression and the like, so we need this to make decisions about modelling. 

We are learning the behavior and limitation of the learning algorithm not just using it, sometimes the model may learn it well, underfit it, overfit the training data, or fail to capture some nonlinear relationships. (write the definitions of these things.)

At last, we will compare the training vs validation/test performance and visualize our predictions and errors. 

** Question: now since it is relatively a smaller amount of dataset we can visually see these things and compare them with our model, what if the data is big, how are we going to approach that?--

We can use some of the follwong methods:
1. Use random sample of data for visualization
2. We can create aggregated plot
3. We can look at distributions and percentiles then finally compare these patters to those that the model learns later. 






Looking at the individual relationships between the variables and the house price, by looking at the scatter plots we can infer the following idea:
Most individual features show little obvious linear relationship with house price, while Square_Footage shows a strong positive linear relationship. Further analysis is needed to determine whether the other features contribute useful information when considered together. 

The beauty of machine learning models is so that this model can help us see if there are things that we can not see
Also, your plots are showing marginal relationships—one feature against price at a time. A machine-learning model can discover relationships that aren't obvious in these individual scatter plots. 
“Before we let the algorithm learn, what can we learn about the relationships in our data ourselves?” To check this we’re going to work on the correlation part, meaning we are trying to understand which features are actually related to the things we are trying to predict. Meaning if there is one specific variable, when this feature changes, does the price of house change with it?
The reason we are doing this is because we want to understand what we are feeding into the model and potentialy catch things like irrelevant features or suspicious relationships. 

Barney@MERCY MINGW64 ~/AI-projects/project-1-supervised-learning
$ python scripts/correlation.py
Correlation with House Price:
House_Price             1.000000
Square_Footage          0.991261
Lot_Size                0.160412
Garage_Size             0.052133
Year_Built              0.051967
Num_Bedrooms            0.014633
Num_Bathrooms          -0.001862
Neighborhood_Quality   -0.007770
Name: House_Price, dtype: float64

There is one overwhelmingly strong individual relationship: Square_Footage ↔ House_Price. 


Step 2: The Learning curve
Okay, now we are moving our dataset to our actual learning part. 
Defining our x and y: which is the input features, which are the information about the houses, then we have our y features which is the target variables which is the house price we want the model to predict. 

              X (features)
        ┌─────────────────────┐
        │ Square_Footage       │
        │ Num_Bedrooms         │
        │ Num_Bathrooms        │
        │ Year_Built           │
        │ Lot_Size             │
        │ Garage_Size          │
        │ Neighborhood_Quality │
        └──────────┬──────────┘
                   │
                   ▼
              ML MODEL
                   │
                   ▼
             House_Price
                   y
We need to split our data: the reason we are doing this is because we can not train and test on the same data. If so then we have no reliable way to tell whether a moel learned a general relationship or simply fit the examples it saw. 
So, we split the data into two, the training data, this is going to be where the model learn from, and then the testing data, this is going where our model is going to be evaluated on. 
We create our modelling.py file now for this purpose. 
$ python scripts/modeling.py
Original dataset: (1000, 8)
Training features: (800, 7)
Testing features: (200, 7)
Training target: (800,)
Testing target: (200,)


When we ran our modeling.py we can see that we have classified or broken our data into two, our training and testing features and targets. 

Yes. Let's pause the code.
What are we doing right now?
We've split our data, and now we want to create a model that can predict house prices.
For now, we're deliberately using only one feature:
Square_Footage
Why only one? Because we want to understand the learning process before introducing seven features at once.
The model we're using
We're starting with linear regression. The model is basically trying to find a straight line that describes the relationship between square footage and price.
Imagine our data looked like this:
House size                         Price
1000 sq ft    ───────────────────→ $200,000
1500 sq ft    ───────────────────→ $250,000
2000 sq ft    ───────────────────→ $300,000
2500 sq ft    ───────────────────→ $350,000
We want the model to discover a rule that approximately says:
"Given the square footage, I can estimate the price."
That rule is represented by a line.
Where does learning come in?
The model's line is controlled by two numbers:
w (weight/slope): controls how much the predicted price changes when square footage changes.
b (bias/intercept): controls where the line starts vertically.
Initially, the model doesn't know what these values should be.
So we start with some initial values.
Then:
1. Model makes predictions
It uses its current line to estimate prices.
2. We compare predictions to the actual prices
For example, if the actual price is $300,000 but the model predicts $100,000, the model is very wrong.
3. We measure how wrong it is
This gives us an error/loss.
4. We use that error to adjust w and b
We change the line so that its predictions become better.
5. Repeat
Prediction → error → adjustment → prediction → error → adjustment...
That repeated process is learning.
So right now, we're at the very beginning:
We have the training data. We need to give the model an initial line so it can make its first predictions.
Once we do that, we can get to the genuinely interesting part: how the model uses its mistakes to learn.
arning
$ python scripts/modeling.py

Initial predictions:
0    0
1    0
2    0
3    0
4    0
Name: Square_Footage, dtype: int64
(.venv) 
Barn


Now the model is predicint everything at zero right and that is like super wrong but we need to turn all those prediction mistakes into one number that tells us how bad the model currently is. 
So we are going to use MSE, mean squared error as our loss function. The idea is that we take each prediction error , we square it and we average all of them. 

So if the value that the model predicted are far from the actual prices, then it will be a very large loss, 

The reason we use MSE is because
When we square them we tend to remove all the negative values. 
We can heavily penalize for large errors and lowly penalize for smaller errors. 

arning
$ python scripts/loss.py
Mean Squared Error: 444390967226.5864
(.venv) 
Barney@MERCY MINGW64 ~/AI-projects/project-1-supervised-learning
$ 

This is the output we got meaning we are very far off, so now what we can do is we are going to adjust the parameters so that the mse becomes smaller and smaller, an for this adjustment we are going to be using Dradient descnset to adjust the weight and the bias. So we make predictions, we calculate the MSE, we calculate the gradient, we update the w and b and we repeat the entire process. 

Exactly. These values tell us the direction and strength of the first update.
dw = -4,090,345,052 → negative, so Gradient Descent will increase w because we subtract the gradient.
db = -1,234,614 → negative, so it will also increase b.
-We multiply the gradients by a learning rate first. The learning rate controls how big each adjustment is.
We need to choose our learning rae based on how sensitive the loss it to parameter changes and the scale of our features. 
For now we are starting with this: 0.00000001 

Wenow need to go into a loop meaning change the GD into a loop, the conditions to when the loop stops are if there are barely any changes happening anymore and we can also give it a fixed number of iterations called epochs. 

We multiply the gradients by a learning rate first. The learning rate controls how big each adjustment is.
We need to choose our learning rate based on how sensitive the loss is to parameter changes and the scale of our features.
For now we are starting with this: 0.00000001
We now need to go into a loop meaning change the GD into a loop, the conditions to when the loop stops are if there are barely any changes happening anymore and we can also give it a fixed number of iterations called epochs.
For each iteration we:
Make predictions using the current w and b.
Calculate the errors between our predictions and the actual values.
Calculate the MSE to see how much error the model currently has.
Calculate the gradients dw and db.
Update w and b using the learning rate.
Repeat this process.
We can also store the loss from each iteration so we can later plot it and see how the model is learning.
When we ran it for 10,000 iterations, the MSE kept decreasing. It decreased very quickly at the beginning and then the improvements became much smaller. This means the model was approaching a point where further updates were barely changing the loss, which is what we expect as Gradient Descent converges.
Our loss went from about 4.44 × 10¹¹ at the beginning to about 1.62 × 10⁹ after 10,000 iterations. The loss did not reach zero, but it became much smaller and eventually started changing very slowly.
rney@MERCY MINGW64 ~/AI-projects/project-1-supervised-learning
$ python scripts/loss.py
Iteration 1000: MSE = 1620269153.9224536, w = 216.49544899699464, b = 0.24636295563286437
Iteration 2000: MSE = 1620265842.2734213, w = 216.49539487865752, b = 0.42834231498250463
Iteration 3000: MSE = 1620262530.6461854, w = 216.49534076049852, b = 0.610321075475362
Iteration 4000: MSE = 1620259219.0407453, w = 216.4952866425176, b = 0.7922992371134004
Iteration 5000: MSE = 1620255907.4570997, w = 216.49523252471465, b = 0.9742767998986042
Iteration 6000: MSE = 1620252595.8952503, w = 216.49517840708995, b = 1.1562537638329369
Iteration 7000: MSE = 1620249284.3551962, w = 216.4951242896433, b = 1.338230128918369
Iteration 8000: MSE = 1620245972.836937, w = 216.4950701723747, b = 1.5202058951568735
Iteration 9000: MSE = 1620242661.340473, w = 216.49501605528417, b = 1.702181062550418
Iteration 10000: MSE = 1620239349.8658044, w = 216.4949619383718, b = 1.8841556311009757
(.venv) 
Barney@MERCY MINGW64 ~/AI-projects/project-1-supervised-learning
$ python scripts/loss.py
Iteration 1: MSE = 444390967226.5864, w = 40.90345052423368, b = 0.012346144017511588
Iteration 2: MSE = 292886962106.7967, w = 74.07883166532162, b = 0.0223940589071166
Iteration 3: MSE = 193223478136.5658, w = 100.98624166794153, b = 0.030577959237587875
Iteration 4: MSE = 127662108789.99548, w = 122.80991597347382, b = 0.03725002150347219
Iteration 5: MSE = 84534044433.74704, w = 140.5103471965308, b = 0.04269588394040447
Iteration 6: MSE = 56163222504.91207, w = 154.8665578443679, b = 0.04714721789192974
Iteration 7: MSE = 37500118817.121086, w = 166.5103862649097, b = 0.050791924012526767
Iteration 8: MSE = 25223018063.891678, w = 175.95429479934668, b = 0.053782402057008925
Iteration 9: MSE = 17146804941.270185, w = 183.61392401777238, b = 0.05624225822212739
Iteration 10: MSE = 11834050756.239452, w = 189.82638568357297, b = 0.05827174524053254
Iteration 11: MSE = 8339175546.144523, w = 194.86509954734626, b = 0.05995217465378165
Iteration 12: MSE = 6040150913.947027, w = 198.95182695988606, b = 0.06134949545514035
Iteration 13: MSE = 4527789543.921928, w = 202.26643092157306, b = 0.06251719660361067
Iteration 14: MSE = 3532916812.937141, w = 204.95479212257337, b = 0.06349866115317795
Iteration 15: MSE = 2878462281.724736, w = 207.1352293706367, b = 0.06432907560603272
Iteration 16: MSE = 2447944165.717841, w = 208.90370697914386, b = 0.06503697852326439
Iteration 17: MSE = 2164737556.817877, w = 210.33805830035564, b = 0.06564551654970402
Iteration 18: MSE = 1978436491.8697512, w = 211.50141128789065, b = 0.06617346313245213
Iteration 19: MSE = 1855882547.7883246, w = 212.4449668525708, b = 0.06663604476842042
Iteration 20: MSE = 1775263205.7232049, w = 213.2102522912388, b = 0.06704561114527995
Iteration 21: MSE = 1722229593.3719501, w = 213.83094896532714, b = 0.06741217866972347
Iteration 22: MSE = 1687342629.5461886, w = 214.33437466805117, b = 0.06774387130453506
Iteration 23: MSE = 1664393028.1220593, w = 214.74268592142843, b = 0.0680472781163672
Iteration 24: MSE = 1649296150.6269617, w = 215.07385311801747, b = 0.06832774327043885
Iteration 25: MSE = 1639365010.0032814, w = 215.34245142482564, b = 0.06858960123525579
Iteration 26: MSE = 1632832032.5731475, w = 215.5603022582522, b = 0.06883636754906539
Iteration 27: MSE = 1628534459.9608397, w = 215.73699356234016, b = 0.06907089354396344
Iteration 28: MSE = 1625707397.8611054, w = 215.8803017885627, b = 0.06929549183729286
Iteration 29: MSE = 1623847677.8509774, w = 215.9965341491065, b = 0.06951203811339872
Iteration 30: MSE = 1622624301.9610193, w = 216.09080620673257, b = 0.06972205367530729
Iteration 31: MSE = 1621819530.6615424, w = 216.16726701835964, b = 0.06992677239955214
Iteration 32: MSE = 1621290128.926452, w = 216.22928174127725, b = 0.07012719504092949
Iteration 33: MSE = 1620941872.832655, w = 216.27957973876119, b = 0.0703241332772166
Iteration 34: MSE = 1620712779.3029287, w = 216.32037470344167, b = 0.07051824543232715
Iteration 35: MSE = 1620562074.1714802, w = 216.35346208523345, b = 0.07071006545013334
Iteration 36: MSE = 1620462935.0753253, w = 216.38029811177745, b = 0.07090002639413578
Iteration 37: MSE = 1620397717.5294793, w = 216.4020638792015, b = 0.07108847950723729
Iteration 38: MSE = 1620354814.5084743, w = 216.41971733393103, b = 0.07127570967046935
Iteration 39: MSE = 1620326590.593059, w = 216.43403543334688, b = 0.07146194794103329
Iteration 40: MSE = 1620308022.9953349, w = 216.445648340844, b = 0.07164738172147397
(.venv) 
Barney@MERCY MINGW64 ~/AI-projects/project-1-supervised-learning
$ 









Once we put Gradient Descent into a loop, we can train the model by repeatedly updating w and b.
At the beginning, the model is far from a good fit, so the gradients are large and the loss changes quickly. As the parameters get closer to a good solution, the updates become smaller and the loss starts to level out. This is what we mean by convergence.
We trained our model for 10,000 iterations. Most of the improvement happened in the early iterations, while later iterations produced much smaller changes.
We then used the final w and b to make predictions on the test set, which the model did not use during training. This lets us see how well the model performs on data it has not seen before.
For our model:
Training MSE ≈ 1.62 × 10⁹
Test MSE ≈ 1.63 × 10⁹
Test RMSE ≈ 40,425
MSE is useful for measuring the loss, but because it is in squared price units, it is difficult to interpret directly.
RMSE is the square root of MSE, so it puts the error back into the same units as the target. In our case, the model has an RMSE of about 40,425, meaning its prediction error is around 40,425 price-units in RMSE terms.
The training and test MSE are fairly close, meaning the model's error on unseen data is similar to its error on the training data.
At this point, we have built and evaluated a simple univariate linear regression model using only Square_Footage. The model is:
ŷ = wx + b
and Gradient Descent was used to learn w and b from the training data.



STEP 3: Adding multiple features
So until now we have just been training our model by a single variable from the 7 features, the next step is to change the learning curve from simple regression with one feature to multiple linear regression. 

When we add multiple features and ran our loss.py we got the following data:
Barney@MERCY MINGW64 ~/AI-projects/project-1-supervised-learning
$ python scripts/loss.py
Iteration 1000: MSE = 1095287003.2574909, w = Square_Footage          199.705359
Num_Bedrooms              0.449805
Num_Bathrooms             0.136879
Year_Built               28.421372
Lot_Size                  0.500128
Garage_Size               0.100699
Neighborhood_Quality      0.083397
dtype: float64, b = 0.01051875541778008
Iteration 2000: MSE = 1095247707.5016112, w = Square_Footage          199.708398
Num_Bedrooms              0.864270
Num_Bathrooms             0.250526
Year_Built               28.415088
Lot_Size                  0.948535
Garage_Size               0.184356
Neighborhood_Quality      0.081812
dtype: float64, b = 0.00668299604767248
Iteration 3000: MSE = 1095208430.7335572, w = Square_Footage          199.708386
Num_Bedrooms              1.278725
Num_Bathrooms             0.364176
Year_Built               28.413708
Lot_Size                  1.396933
Garage_Size               0.268014
Neighborhood_Quality      0.080241
dtype: float64, b = 0.0028496862248650937
Iteration 4000: MSE = 1095169156.78144, w = Square_Footage          199.708373
Num_Bedrooms              1.693163
Num_Bathrooms             0.477823
Year_Built               28.412329
Lot_Size                  1.845316
Garage_Size               0.351669
Neighborhood_Quality      0.078669
dtype: float64, b = -0.0009836399605273405
Iteration 5000: MSE = 1095129885.6450546, w = Square_Footage          199.708361
Num_Bedrooms              2.107584
Num_Bathrooms             0.591468
Year_Built               28.410950
Lot_Size                  2.293685
Garage_Size               0.435322
Neighborhood_Quality      0.077097
dtype: float64, b = -0.004816982683063133
Iteration 6000: MSE = 1095090617.324194, w = Square_Footage          199.708348
Num_Bedrooms              2.521988
Num_Bathrooms             0.705111
Year_Built               28.409571
Lot_Size                  2.742039
Garage_Size               0.518973
Neighborhood_Quality      0.075524
dtype: float64, b = -0.008650341942193948
Iteration 7000: MSE = 1095051351.8186524, w = Square_Footage          199.708336
Num_Bedrooms              2.936375
Num_Bathrooms             0.818752
Year_Built               28.408192
Lot_Size                  3.190378
Garage_Size               0.602621
Neighborhood_Quality      0.073951
dtype: float64, b = -0.01248371773735879
Iteration 8000: MSE = 1095012089.1282237, w = Square_Footage          199.708323
Num_Bedrooms              3.350745
Num_Bathrooms             0.932390
Year_Built               28.406814
Lot_Size                  3.638702
Garage_Size               0.686267
Neighborhood_Quality      0.072377
dtype: float64, b = -0.016317110067996757
Iteration 9000: MSE = 1094972829.252702, w = Square_Footage          199.708311
Num_Bedrooms              3.765098
Num_Bathrooms             1.046027
Year_Built               28.405435
Lot_Size                  4.087012
Garage_Size               0.769912
Neighborhood_Quality      0.070803
dtype: float64, b = -0.020150518933546874
Iteration 10000: MSE = 1094933572.1918807, w = Square_Footage          199.708298
Num_Bedrooms              4.179434
Num_Bathrooms             1.159661
Year_Built               28.404056
Lot_Size                  4.535307
Garage_Size               0.853554
Neighborhood_Quality      0.069228
dtype: float64, b = -0.02398394433344843
Test MSE: 1100359070.2294674
Test RMSE: 33171.66064925703
(.venv) 
Barney@MERCY MINGW64 ~/AI-projects/project-1-supervised-learning
$ 

—-
Exactly. Yes, in this experiment, adding all 7 features improved the model.
Compare the test set, since that's the data we use to evaluate the final model:
Model
Test MSE
Test RMSE
Square_Footage only
~1.634B
~40,425
All 7 features
~1.100B
~33,172

So both MSE and RMSE decreased substantially.
That tells us:
The multivariate model's predictions are closer to the actual house prices on unseen test data than the one-feature model's predictions.

Test MSE: 1098716550.2546504
Test RMSE: 33146.89352344576

Final model parameters:
Square_Footage: 199.70779800122455
Num_Bedrooms: 20.73896045835205
Num_Bathrooms: 5.703178360167412
Year_Built: 28.348949207994114
Lot_Size: 22.455047263997784
Garage_Size: 4.197419486432673
Neighborhood_Quality: 0.005873055106871216
Bias: -0.17733451206124431
(.venv) 
Barney@MERCY MINGW64 ~/AI-projects/project-1-supervised-learning
After seeing that the one-feature model was able to learn a relationship between Square_Footage and House_Price, we then trained the model using all 7 features.
The model learned a weight for each feature and a bias:
Square_Footage: 199.707798
Num_Bedrooms: 20.738960
Num_Bathrooms: 5.703178
Year_Built: 28.348949
Lot_Size: 22.455047
Garage_Size: 4.197419
Neighborhood_Quality: 0.005873
Bias: -0.177335

The model now makes predictions by multiplying each feature by its learned weight, adding them together, and then adding the bias.
The final model achieved:
Test MSE: 1098716550.2546504
Test RMSE: 33146.89352344576

This means that using all 7 features gave us a lower test error than using only Square_Footage. The model was therefore able to use information from the other features to make better predictions on the unseen test data.
The weights also show that the model does not treat every feature equally. However, we cannot directly say that a larger weight means a feature is more important because the features are measured on very different scales.



We then looked at regularization.
Regularization is used to discourage the model from relying too heavily on large weights. We used L2 regularization, which adds a penalty based on the squared weights to the training loss.
The regularized loss became:
[
J_{reg} = MSE + \lambda \sum_j w_j^2
]
The gradient for each weight also gained an additional regularization term:
[
dw = \frac{2}{n}X^T(errors) + 2\lambda w
]
We did not regularize the bias, so the bias gradient stayed the same.
We first tried a small regularization strength:
lambda = 1
The weights barely changed and the test performance was almost identical to the unregularized model.
This showed us that when lambda is too small, the regularization penalty has very little effect on the model.
We then increased lambda to see what would happen.
With a much larger value:
lambda = 100000
the weights changed substantially:
Square_Footage: 199.708298 → 191.036340
Num_Bedrooms: 4.179434 → 0.260566
Num_Bathrooms: 1.159661 → 0.092130
Lot_Size: 4.535307 → 0.289710
Garage_Size: 0.853554 → 0.064335
Some weights became much smaller, showing the effect of L2 regularization.
However, the test performance became worse:
Unregularized:
Test MSE: 1100359070.2294674
Test RMSE: 33171.66064925703

Regularized:
Test MSE: 1259815289.1083806
Test RMSE: 35493.87678330419
This showed us that making the regularization strength too large can constrain the model too much and cause it to lose useful information from the features.
So from these experiments, we saw three different situations:
A small lambda → very little effect.
A sufficiently strong lambda → weights are pushed toward zero.
A very large lambda → the model can become too constrained and perform worse.
We also plotted the training loss to see how the model changed during learning. Since the loss decreased very quickly during the first few iterations and then changed much more slowly, we plotted the first 20 iterations to make the learning process easier to see.
Finally, we plotted the actual house prices against the predicted house prices. The diagonal line represents perfect predictions, so points closer to this line represent predictions that are closer to the actual values.

CONCLUSION
Project Conclusion
In this project, we built a linear regression model from scratch to predict house prices.
We first explored the dataset and looked at the relationships between the features and House_Price. We found that Square_Footage had a very strong relationship with the target, while the other individual features had much weaker linear correlations.
We then trained a model using only Square_Footage and used MSE as our loss function. We implemented Gradient Descent ourselves and saw how the model gradually changed its parameters to reduce the loss. The loss decreased very quickly at the beginning and then improved more slowly as the model approached convergence.
After that, we extended the model to use all 7 features. The test RMSE decreased from about 40,425 with one feature to about 33,147 with all 7 features. This showed us that features that have weak individual correlations can still provide useful information when they are used together in a multivariable model.
We then implemented L2 regularization from scratch. We saw that a small regularization strength such as lambda = 1 had almost no visible effect, while a very large value such as lambda = 100000 pushed several weights toward zero but also made the test performance worse.
This helped us understand that regularization is not simply about making the weights smaller. The regularization strength has to be appropriate for the problem. If it is too weak, it has little effect, and if it is too strong, it can constrain the model too much.
Overall, this project helped us understand the complete basic workflow of supervised learning:
Dataset
    ↓
Explore the data
    ↓
Choose features and target
    ↓
Split into training and testing data
    ↓
Make predictions
    ↓
Calculate loss
    ↓
Calculate gradients
    ↓
Update parameters with Gradient Descent
    ↓
Repeat until convergence
    ↓
Evaluate on unseen test data
    ↓
Experiment with regularization

Most importantly, we did not just use a machine learning library to train the model. We implemented the main learning process ourselves, which helped us understand what is actually happening inside a linear regression model.


Yes — that is actually one of the most important things we learned from this project.
There are a few research questions we can ask after training the model:
Did the model learn the relationship we observed during exploration?
Yes. Square_Footage had an extremely strong relationship with House_Price, and the trained model gave it a large weight. So the pattern we saw visually and through correlation was also reflected in what the model learned.
Did the model learn anything that wasn't obvious during exploration?
Yes. This is probably the more interesting part.
When we looked at the features individually, several of them appeared to have very weak relationships with house price. For example, Num_Bedrooms, Num_Bathrooms, and Garage_Size didn't show strong individual linear relationships.
But when we gave all seven features to the model, the test RMSE improved from about 40,425 → 33,147.
That means the additional features were providing useful information when considered together, even though their individual relationships weren't obvious to our eyes.
So did the model "discover" something we couldn't see?
In a sense, yes — but with an important distinction.
The model didn't magically discover causality or prove that those features cause house prices to change. What it discovered during training were parameter values that helped predict the target when the features were considered together.
That's one of the powerful things about machine learning:
We can observe some patterns ourselves, but the model can learn combinations of patterns that are difficult to see directly.
And this connects really nicely to why we did the exploration before modeling.
We first asked:
"What can we understand about the data ourselves?"
Then we asked:
"What can the model learn from the data that we might not see immediately?"
That's a much more meaningful ML workflow than simply throwing the dataset into an algorithm and looking at the final accuracy.
So our final research questions could be:
Did the model learn the strong Square_Footage → House_Price relationship we observed?
Did the additional features improve prediction even when their individual relationships appeared weak?
Did Gradient Descent actually converge toward parameters that reduced the prediction error?
What happened when we introduced regularization?
How did changing the regularization strength affect the learned weights and test performance?
Did the model generalize from the training data to unseen test data?
And yes, that first two questions are especially valuable for our project. They connect our human exploration → model learning → evaluation, instead of treating those as three unrelated steps.

—
RESEARCH QUESTION AND ANSWER FROM OUR EVALUATION

Exactly. The research questions aren't just things we ask — we use the results of our experiments to answer them.
For our project, we can answer them like this:
1. Did the model learn the relationship we observed?
Yes.
During exploration, we found that Square_Footage had an extremely strong positive relationship with House_Price (r ≈ 0.99).
When we trained the model, Square_Footage received a large learned weight:
w ≈ 199.7
So the relationship we observed during EDA was also reflected in the model's learned parameters.

2. Did the other features provide useful information even though their individual relationships were weak?
Yes.
Individually, several features looked weak when we examined their correlations and scatter plots.
But when we added all 7 features:
Square Footage only: RMSE ≈ 40,425
All 7 features: RMSE ≈ 33,147
The substantial decrease in test error shows that the additional features contributed useful predictive information when considered together.
This is probably our most interesting finding.

3. Did Gradient Descent actually learn the model?
Yes.
The training MSE decreased substantially during training and then began changing much more slowly.
That showed that Gradient Descent was successfully adjusting the weights and bias to reduce the prediction error and was approaching convergence.

4. Did the model generalize to unseen data?
Yes, to some extent.
We trained the parameters using the training set and then evaluated those fixed parameters on the test set.
The test RMSE was around 33,147, showing that the model could make reasonably useful predictions on data it had not used to update its parameters.

5. What happened when we added regularization?
We found that the strength of regularization mattered.
λ = 1 → almost no noticeable effect.
λ = 100,000 → weights were strongly constrained and test RMSE became worse (~35,494).
So our experiment demonstrated that regularization can control the size of weights, but too much regularization can hurt the model's ability to fit the data.

So the overall pattern is:
Question → Experiment → Observation → Answer
And that's what makes this feel more like an actual ML investigation rather than just:
"I trained a model and got an RMSE."

