import numpy as np

class Node(): # Class to define & control tree nodes
      def __init__(self):
        self.split    = None
        self.feature  = None
        self.left     = None
        self.right    = None
        self.leaf_value = None

class DecisionTree():

    def __init__(self, max_depth: int=None, min_samples_split: int=2):
        self.tree              = None
        self.max_depth         = max_depth   # max depth that the tree can grow
        self.min_samples_split = min_samples_split  # minimum number of samples required to split a node
    
    
        
    def grow(self, node: Node, D: np.array, level: int):  #function to grow a tree during training
       
        # to check if we are in a leaf node?
        depth = (self.max_depth is None) or (self.max_depth >= (level+1))
        msamp = (self.min_samples_split <= D.shape[0])
        n_cls = np.unique(D[:,-1]).shape[0] != 1
        
        # proceed if it is not a leaf node
        if depth and msamp and n_cls:
        
            # All the function parameters are initialized
            ip_node = None
            feature = None
            split   = None
            left_D  = None
            right_D = None
            # iteration through all the possible feature and split 
            for f in range(D.shape[1]-1):
                for s in np.unique(D[:,f]):
                    # for the recent feature and split combination, split the dataset
                    D_l = D[D[:,f]<=s]
                    D_r = D[D[:,f]>s]
                    # ensure we have non-empty arrays
                    if D_l.size and D_r.size:
                        # calculate the impurity
                        ip  = (D_l.shape[0]/D.shape[0])*self.impurity(D_l) + (D_r.shape[0]/D.shape[0])*self.impurity(D_r)
                        # now update the impurity and choice of (f,s)
                        if (ip_node is None) or (ip < ip_node):
                            ip_node = ip
                            feature = f
                            split   = s
                            left_D  = D_l
                            right_D = D_r
            # set the current node's parameters
            # node.set_params(split,feature)
            node.split = split
            node.feature = feature

            # declare child nodes
            left_node  = Node()
            right_node = Node()
            # node.set_children(left_node,right_node)
            node.left = left_node
            node.right = right_node

            # investigate child nodes
            self.grow(node.left,left_D,level+1)
            self.grow(node.right,right_D,level+1)
                        
        # is a leaf node
        else:
            
            # set the node value & return
            node.leaf_value = self.leaf_value(D)
            return
    
    
    
    def traverse(self, node: Node, Xrow: np.array):  # Fucntion to traverse a trained tree gives leaf value of Xrow as o/p
              
        # check if we're in a leaf node?
        if node.leaf_value is None:
            # get parameters at the node
            # (s,f) = node.get_params()
            # decide to go left or right?
            if (Xrow[node.feature] <= node.split):
                return(self.traverse(node.left,Xrow))
            else:
                return(self.traverse(node.right,Xrow))
        else:
            # return the leaf value
            return(node.leaf_value)
        
    
    def train(self, Xin: np.array, Yin: np.array): # Fucntion to train the cart model
               
        # prepare the input data
        D = np.concatenate((Xin,Yin.reshape(-1,1)),axis=1)
        # set the root node of the tree
        self.tree = Node()
        # build the tree
        self.grow(self.tree,D,1)
        
    def predict(self, Xin: np.array):  # fucntion makes prediction from trained cart model
        # iterate through the rows of Xin
        p = []
        for r in range(Xin.shape[0]):
            p.append(self.traverse(self.tree,Xin[r,:]))
        # return predictions
        return(np.array(p).flatten())
    


class DecisionTreeClassify(DecisionTree):
    
    def __init__(self, max_depth: int=None, min_samples_split: int=2, loss: str='gini'):
       
        DecisionTree.__init__(self,max_depth,min_samples_split)
        self.loss = loss   
    
    def gini(self, D: np.array):
               
        # initialize the output
        G = 0
        # iterate through the unique classes
        for c in np.unique(D[:,-1]):
            # compute p for the current c
            p = D[D[:,-1]==c].shape[0]/D.shape[0]
            # compute term for the current c
            G += p*(1-p)
        # return gini impurity
        return(G)
    
    def entropy(self, D: np.array):  #shannon entropy loass function for data
               
        # initialize the output
        H = 0
        # iterate through the unique classes
        for c in np.unique(D[:,-1]):
            # compute p for the current c
            p = D[D[:,-1]==c].shape[0]/D.shape[0]
            # compute term for the current c
            H -= p*np.log2(p)
        # return entropy
        return(H)
    
    def impurity(self, D: np.array):     
        # use the selected loss function to calculate the node impurity
        ip = None
        if self.loss == 'gini':
            ip = self.gini(D)
        elif self.loss == 'entropy':
            ip = self.entropy(D)
        # return results
        return(ip)
    
    def leaf_value(self, D: np.array): #gives mode value of data as output
        
        # Find unique elements and their counts
        unique_elements, counts = np.unique(D[:,-1], return_counts=True)
    
        # Find the index of the maximum count
        max_count_index = np.argmax(counts)
    
        # Return the mode(s)
        mode_values = unique_elements[counts == counts[max_count_index]]
        return mode_values[0]       



class DecisionTreeRegression(DecisionTree):

    
    def __init__(self, max_depth: int=None, min_samples_split: int=2, loss: str='mse'):
              
        DecisionTree.__init__(self,max_depth,min_samples_split)
        self.loss = loss   
    
    def mse(self, D: np.array): #output mean square error over the data
        
        # compute the mean target for the node
        y_m = np.mean(D[:,-1])
        # compute the mean squared error wrt the mean
        E = np.sum((D[:,-1] - y_m)**2)/D.shape[0]
        # return mse
        return(E)
    
    def mae(self, D: np.array): #gives mean absolute error over data
        
        # compute the mean target for the node
        y_m = np.mean(D[:,-1])
        # compute the mean absolute error wrt the mean
        E = np.sum(np.abs(D[:,-1] - y_m))/D.shape[0]
        # return mae
        return(E)
    
    def impurity(self, D: np.array): # gives impurity metric as output
                   
        # use the selected loss function to calculate the node impurity
        ip = None
        if self.loss == 'mse':
            ip = self.mse(D)
        elif self.loss == 'mae':
            ip = self.mae(D)
        # return results
        return(ip)
    
    def leaf_value(self, D: np.array) -> float:   #mean of data is given as output
        
        return(np.mean(D[:,-1]))
    








    


    