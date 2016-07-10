# social_network_diffusion
  The code implements the model proposed in "Learning Social Network Embeddings for Predicting Information Diffusion - Simon Bourigault, Cedric Lagnier, Sylvain Lamprier, Ludovic Denoyer, Patrick Gallinari, Université Pierre et Marie Curie" published at WSDM 2014. During development, I have refered a lot to original implementation by Ludovic Denoyer
  
  Dependencies:
  Tensorflow.
  
  Data formats.
  The software needs two input files: a training set of cascades and a testing set. 
  For CDK model, the format of each file is the following:
  one line for each cascade
  each column is [name of the user],[timestamp of the contamination]
  Typically the timestamp of the forst column is one since it corresponds to the source of the cascade.
  
  For CSDK model, the format of each file is similar to above but requires a content for cascades.
  Specifically:
  one line for each cascade
  first column is content, followed by [name of the user],[timestamp of the contamination] pairs.
  
  Only users that appear at least once in both the train and test files are kept. For CSDK, content that appear at least once in both the train and test files are kept.
  
  Contact:songweipingfight@gmail.com
