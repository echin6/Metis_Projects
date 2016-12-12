##Project 5:  Kojak  
##Topic:      Neural Restoration: Applying A.I. to Art Restoration
 
--

###Problem:
Can a machine learn to understand and appreciate art in a general sense?  How can one teach a machine to understand higher-order
functions?

###Goal:

The goal of this project is to restore unseen damaged images/photographs with a content-aware neural net.

###Approach:
~200,000 painting images were scraped and augmented from Artsy.net for model training.  1,000 images were set aside each for
the validation and test set.  The training set includes a wide variety of style and subject matters from the turn of the 18th
century until now.

Using a deep convolutional generative adversarial networks, two neural nets are trained simultaneously to reconstruct art images
from artificially "masked" images.  

- the generator generates images that feed into the discriminator for feedback

- the discriminator takes in real images and generated images and tries to tell them apart.  It gets rewarded for assigning
high probabilities to real images and low probabilities to "fake images".  No other heuristics cost functions are needed.

In convergence the generator creates such realistic images to the point the discriminator can no longer distinguish the real images
from the generated images.  


###Findings:
The neural net (DCGAN) generated very satisfying results that went beyond my own expectations.  The machine learned to fill in large 
patches of missing pixels in a high resolution manner, and showed tremendous common sense in adapting to a wide range of styles and 
subject matters.

The only drawback of the model is the high degree of difficulty for it to be trained properly.


###Recommendations:
The DEGAN architecture has numerous real-world applications in predictive analytics and feature extractions and is widely used by 
companies like Facebook and google.  It has shown great promise in training machines to learn higher-ordered functions like 
reasoning, planning and prediction.


---

My code is contained in the jupyter notebook in my [GitHub](https://github.com/echin6/my_recent_projects) repo 


PDF slides of presentation have also been uploaded.
