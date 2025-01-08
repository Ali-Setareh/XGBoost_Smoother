create_S_from_tree_L1 =  function(current_tree_indices_train, current_tree_indices_test,
                                  current_tree_leaf_scores,y_train,lambda,alpha) {
  
  S_train = outer(current_tree_indices_train, current_tree_indices_train, FUN = "==")
  S_train = S_train/(rowSums(S_train)+lambda)
  
  S_test = outer(current_tree_indices_test, current_tree_indices_train, FUN = "==")
  S_test = S_test/(rowSums(S_test)+lambda)
  
  # Get unique leaves
  all_leaves = unique(current_tree_indices_train)
  
  n_train = length(current_tree_indices_train)
  n_test = length(current_tree_indices_test)
  
  L1_corr_train = matrix(0, nrow = n_train, ncol = n_train)
  L1_corr_test = matrix(0, nrow = n_test, ncol = n_train)
  
  # For each leaf
  for (leaf in all_leaves) {
    
    leaf_weight = current_tree_leaf_scores[Node==leaf]$Quality
    
    
    
    # Indices of training data in this leaf
    indices_in_leaf = which(current_tree_indices_train == leaf)
    indices_in_leaf_test = which(current_tree_indices_test == leaf)
    
    #print(dim(S_test[indices_in_leaf_test, indices_in_leaf]))
    
    # Number of data points in the leaf
    n_leaf = length(indices_in_leaf)
    
    # Compute G_j (Gradient)
    #G_j = sum(prev_tree_predictions_train[indices_in_leaf] - y_train[indices_in_leaf])
    
    
    
    # Apply ThresholdL1
    if (sign(leaf_weight) == 1) {
      #G_tilde_j = G_j - alpha
      
      # Step 2: Compute the update values for each index in indices_in_leaf
      #update_values <- -alpha / ((n_leaf + lambda) * y_train[indices_in_leaf])
      # update_values is a vector of length n_leaf
      
      # Step 3: Update the submatrix of S_train
      # Add update_values[i] to every element in row i of the submatrix
      
      for (i in indices_in_leaf) {
        
        L1_corr_train[i,i] = - (alpha / ((n_leaf + lambda) * y_train[i]))
        
      }
      
      #S_train[indices_in_leaf, indices_in_leaf] <- sweep(
       # S_train[indices_in_leaf, indices_in_leaf], 
        #MARGIN = 1, 
        #STATS = update_values, 
        #FUN = "+"
      #)
      
      if (length(indices_in_leaf_test) > 0) {
        
        for (i in indices_in_leaf_test) {
          
          L1_corr_test[i,indices_in_leaf[1]] = - (alpha / ((n_leaf + lambda) * y_train[indices_in_leaf[1]]))
          
        }
      }
      
      
    } else if (sign(leaf_weight) == -1) {
      #G_tilde_j = G_j + alpha
      
      # Step 2: Compute the update values for each index in indices_in_leaf
      #update_values <- alpha / ((n_leaf + lambda) * y_train[indices_in_leaf])
      # update_values is a vector of length n_leaf
      
      # Step 3: Update the submatrix of S_train
      # Add update_values[i] to every element in row i of the submatrix
      for (i in indices_in_leaf) {
        L1_corr_train[i,i] = (alpha / ((n_leaf + lambda) * y_train[i]))
      }
      
      #S_train[indices_in_leaf, indices_in_leaf] <- sweep(
      # S_train[indices_in_leaf, indices_in_leaf], 
      #MARGIN = 1, 
      #STATS = update_values, 
      #FUN = "+"
      #)
      
      if (length(indices_in_leaf_test) > 0) {
        
        for (i in indices_in_leaf_test) {
          
          L1_corr_test[i,indices_in_leaf[1]] =  (alpha / ((n_leaf + lambda) * y_train[indices_in_leaf[1]]))
          
        }
        
      }
      
    } else if(sign(leaf_weight) == 0) {
      #G_tilde_j = 0
      
      S_train[indices_in_leaf, ] = 0
      # For test data
      indices_in_leaf_test = which(current_tree_indices_test == leaf)
      if (length(indices_in_leaf_test) > 0) {
        S_test[indices_in_leaf_test, ] = 0
      }
    }
  
  }
  
  S_train = S_train + L1_corr_train
  S_test = S_test + L1_corr_test
  
  
  return(list(S_train = S_train, S_test = S_test))
}

## Extract S for a single boosted tree

create_S_from_single_boosted_tree_L1 = function(current_tree_indices_train, current_tree_indices_test,
                                                current_tree_leaf_scores,y_train,S_gb_prev,lambda,alpha){
  
  S  = create_S_from_tree_L1(current_tree_indices_train, current_tree_indices_test,
                          current_tree_leaf_scores,y_train,lambda,alpha)
  
  
  if (is.null(S_gb_prev)){
    
    # first tree: just normal tree
    return (list(S_train = S$S_train, S_test = S$S_test))
  }
  
  n_train = length(current_tree_indices_train)
  n_test = length(current_tree_indices_test)
  
  all_nodes = unique(current_tree_indices_train)
  n_nodes = length(all_nodes)
  
  
  node_corrections = matrix(0, nrow = n_nodes, ncol = n_train)
  S_train_correction = matrix(0, nrow = n_train, ncol = n_train)
  S_test_correction = matrix(0, nrow = n_test, ncol = n_train)
  
  
  
  
  
  for (i in 1:length(all_nodes)) {
    
    n = all_nodes[i]
    leaf_weight = current_tree_leaf_scores[Node==n]$Quality
    
    if (leaf_weight!=0){
      
    
    
    # Create correction matrix
    leaf_id_train = current_tree_indices_train == n
    node_corrections[i, ] = colSums(S_gb_prev[leaf_id_train, , drop = FALSE]) /(sum(leaf_id_train)+lambda)
    
    
    S_train_correction[leaf_id_train, ] = matrix(rep(node_corrections[i, ], sum(leaf_id_train)), nrow = sum(leaf_id_train), byrow = TRUE)
    
    leaf_id_test = current_tree_indices_test == n
    S_test_correction[leaf_id_test, ] = matrix(rep(node_corrections[i, ], sum(leaf_id_test)), nrow = sum(leaf_id_test), byrow = TRUE)
    
    }
  }
  
  
  
  
  #cat("Dimensions of S_train:", dim(S$S_train), "\n")
  #cat("Dimensions of S_train_correction:", dim(S_train_correction), "\n")
  
  
  S_train = S$S_train - S_train_correction
  S_test = S$S_test - S_test_correction
  
  
  
  
  return(list(S_train = S_train, S_test = S_test))
  
}

create_S_from_gbtregressor_L1 = function(model,
                                         leaf_indices_train,
                                         leaf_indices_test,
                                         tree_leaf_scores,
                                         y_train,
                                         lambda,
                                         alpha,
                                         eta) {
  
  n_train = nrow(leaf_indices_train)
  n_test = nrow(leaf_indices_test)
  
  number_of_trees = ncol(leaf_indices_train)
  
  S_train_curr = matrix(0, nrow = n_train, ncol = n_train)
  S_test_curr = matrix(0, nrow = n_test, ncol = n_train)
  
  
  lr = model$params$eta
  lambda = model$params$lambda
  
  # Initialize predictions
  
  
  #predictions_train = rep(0, n_train)
  #predictions_test = rep(0, n_test)
  
  
  #predictions_train = matrix(NA, nrow = n_train, ncol = number_of_trees+1)  # n columns for predictions
  #predictions_test = matrix(NA, nrow = n_test, ncol = number_of_trees+1)  # n columns for predictions
  
  #predictions_train[,1] = 0
  #predictions_test[,1] = 0
  
  
  # Loop through iteration ranges and predict
  #for (i in 1:number_of_trees) {
    # Predict using iteration range c(1, i + 1)
    #predictions_t = predict(model, dtrain, iterationrange = c(1, i + 1))
    #predictions_tst = predict(model, dtest, iterationrange = c(1, i + 1))
    
    # Store predictions in the ith column of the matrix
    #predictions_train[, i+1] = predictions_t
    #predictions_test[, i+1] = predictions_tst
  #}
  
  
  
  
  
  
  
  for (t in 1:number_of_trees) {
    
    cat(sprintf("\r Processing tree: %d/%d", t, number_of_trees))
    
    current_tree_indices_train = leaf_indices_train[, t]
    current_tree_indices_test = leaf_indices_test[, t]
    
    #prev_tree_predictions_train = predictions_train[, t]
    #prev_tree_predictions_test = predictions_test[, t]
    
    current_tree_leaf_scores = tree_leaf_scores[Tree == t-1 & Feature == "Leaf",.(Node,Quality)]
    
    
    S = create_S_from_single_boosted_tree_L1(current_tree_indices_train,current_tree_indices_test,
                                             current_tree_leaf_scores,
                                             y_train,
                                             if(t == 1) NULL else S_train_curr , lambda , alpha)
    
    
    S_train_curr = S_train_curr + lr * S$S_train
    S_test_curr = S_test_curr + lr * S$S_test
    
    
    
    # Update predictions
    #predictions_train = result$predictions_train
    #predictions_test = result$predictions_test
    
  }
  
  return(list(S_train = S_train_curr, S_test = S_test_curr))
}

get_xgboost_weights_L1 = function(model, dtrain, dtest) {
  
  
  leaf_indices_train = predict(model, dtrain, predleaf = TRUE)
  leaf_indices_test = predict(model, dtest, predleaf = TRUE)
  
  tree_leaf_scores = xgb.model.dt.tree(model = model)[Feature == "Leaf"]
  
  
  y_train = getinfo(dtrain, "label")
  
  # Get parameters
  params = model$params
  lambda = ifelse(!is.null(params$lambda), params$lambda, 0)
  alpha = ifelse(!is.null(params$alpha), params$alpha, 0)
  eta = ifelse(!is.null(params$eta), params$eta, 0.3)
  
  smoothers = create_S_from_gbtregressor_L1(model,
                                            leaf_indices_train,
                                            leaf_indices_test,
                                            tree_leaf_scores,
                                            y_train,
                                            lambda,
                                            alpha,
                                            eta)
  
  return(list(S_train = smoothers$S_train, S_test = smoothers$S_test))
}
