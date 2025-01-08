## Extract S for a single tree

create_S_from_tree = function(tree_indices,n, lambda,tree_indices_test = NULL){
  
  
    
  
  #nodes = matrix(tree_indices, ncol = 1)
  
  
  
  #nodes_rep = rep(nodes, times = n)
  
  # Step 2: Create the expanded nodes_train matrix by replicating the single column n_train times
  #nodes_exp = matrix(nodes_rep, nrow = n, byrow = TRUE)
  
  # Step 3: Create a boolean matrix S_train by comparing each element of nodes_train_exp with the replicated nodes_train vector
  #S = (nodes_exp) == t(nodes_exp)
  #S = sweep(S, 1, rowSums(S)+lambda, FUN = "/")
  
  nodes_train = c(tree_indices)
  
  if (is.null(tree_indices_test )){
  
  S = outer(nodes_train, nodes_train, FUN = "==")
  
  # Normalize rows of S_train
  S = S / (rowSums(S)+lambda)
  
  } else {
    
    
    nodes_test = c(tree_indices_test)
    # Step 2: Create S_test by comparing nodes_test with nodes_train
    S = outer(nodes_test, nodes_train, FUN = "==")
    # Normalize rows of S_test
    S = S / (rowSums(S)+lambda)
    
  }
  
  print(dim(S))
  
  
  return(S)
}



## Extract S for a single boosted tree

create_S_from_single_boosted_tree = function(tree_indices,S_gb_prev,num_samples,lambda,test_tree = NULL){
  
  S  = create_S_from_tree(tree_indices,num_samples,lambda,test_tree)
  
  if (is.null(S_gb_prev)){
    
    # first tree: just normal tree
    return (S)
  }
  
  all_nodes = unique(tree_indices)
  n_nodes = length(all_nodes)
  
  
  node_corrections = matrix(0, nrow = n_nodes, ncol = num_samples)
  
  if (is.null(test_tree)){
    
    S_correction = matrix(0, nrow = num_samples, ncol = num_samples)
    
    for (i in 1:length(all_nodes)) {
      
      n = all_nodes[i]
      
      # Create correction matrix
      leaf_id = tree_indices == n
      node_corrections[i, ] = colSums(S_gb_prev[leaf_id, , drop = FALSE]) /(sum(leaf_id)+lambda)
      
      
      S_correction[leaf_id, ] = matrix(rep(node_corrections[i, ], sum(leaf_id)), nrow = sum(leaf_id), byrow = TRUE)
    }
    
    
  }else{
    
    S_correction = matrix(0, nrow = length(test_tree), ncol = num_samples)
    
    for (i in 1:length(all_nodes)) {
      
      n = all_nodes[i]
      
      # Create correction matrix
      leaf_id = test_tree == n
      node_corrections[i, ] = colSums(S_gb_prev[leaf_id, , drop = FALSE]) /(sum(leaf_id)+lambda)
      
      
      S_correction[leaf_id, ] = matrix(rep(node_corrections[i, ], sum(leaf_id)), nrow = sum(leaf_id), byrow = TRUE)
    }
    
  }
  
  
  
  
  S = S - S_correction
  
  
  return(S)
  
  
}


create_S_from_single_boosted_tree_compact = function(tree_indices,S_gb_prev,num_samples,lambda){
  
  S  = create_S_from_tree(tree_indices,num_samples,lambda)
  
  if (is.null(S_gb_prev)){
    
    # first tree: just normal tree
    return (S)
  }
  
  S = S - S %*% S_gb_prev # equivalent but very slow
  
}


# put everything together 

create_S_from_gbtregressor = function(model,leaf_indices,tree_indices_test = NULL,output_dir = NULL, save_output = FALSE, compact = FALSE){
  
  if (compact==FALSE){
    
    if (save_output){
      # Check if the directory exists, and create it if it doesn't
      if (!file.exists(output_dir)) {
        dir.create(output_dir, recursive = TRUE)
      }
    }
  n = nrow(leaf_indices)
  
  if(is.null(tree_indices_test)){
    
    num_row = n
    
  } else{
    
    num_row = nrow(tree_indices_test)
  }
  
  S_curr = matrix(0, nrow = num_row, ncol = n)
  
  lambda = model$params$lambda
  lr = model$params$eta
  
  for (col in 1:ncol(leaf_indices)) {
    
    cat("\rProcessing tree:", col)
    flush.console()
    
    current_tree = leaf_indices[, col]
    if(!is.null(tree_indices_test)){
      
      current_tree_test = tree_indices_test[, col]
      S = create_S_from_single_boosted_tree(current_tree,if(col == 1) NULL else S_curr ,n, lambda,test_tree=current_tree_test)
      
    }else{
      
      S = create_S_from_single_boosted_tree(current_tree,if(col == 1) NULL else S_curr ,n, lambda,test_tree=NULL)
      
    }
    
    
    print(dim(S))
    
    #print(dim(S_curr))
    S_curr = S_curr + lr * S
    
    if (save_output) {
      # Save the current matrix to an RDS file
      output_filename = paste(output_dir, "/S_curr_iteration_", col, ".rds", sep = "")
      saveRDS(S_curr, file = output_filename)
    }
  }
  
  return (S_curr)
  } 
  
  else if (compact==TRUE){
    
    if (save_output){
      # Check if the directory exists, and create it if it doesn't
      if (!file.exists(output_dir)) {
        dir.create(output_dir, recursive = TRUE)
      }
    }
    
    n = nrow(leaf_indices)
    lambda = model$params$lambda
    S_curr = matrix(0, nrow = n, ncol = n)
    lr = model$params$eta
    for (col in 1:ncol(leaf_indices)) {
      
      cat("\rProcessing tree:", col)
      flush.console()
      
      current_tree = leaf_indices[, col]
      S = create_S_from_single_boosted_tree_compact(current_tree,if(col == 1) NULL else S_curr ,n, lambda)
      S_curr = S_curr + lr * S
      
      
      
      if (save_output) {
        # Save the current matrix to an RDS file
        output_filename = paste(output_dir, "/S_curr_iteration_", col, ".rds", sep = "")
        
        
        
        saveRDS(S_curr, file = output_filename)
      }
    }
    
    return (S_curr)
    
  }
  
 
}

# Metrics

compute_metrics_from_S = function(S, y) {
  # compute predictions
  y_pred = S %*% y
  
  # compute MSE
  mse = mean((y - y_pred)^2)
  
  # compute accuracy
  acc = mean((y_pred > 0) == (y > 0))
  
  # compute trace metric
  eff_p_tr = sum(diag(S))
  
  # compute l2-norm
  l2_norm = mean(sqrt(rowSums(S^2)))
  
  # compute squared l2-norm
  l2_norm_sq = mean(rowSums(S^2))
  
  return(list(mse = mse, acc = acc, eff_p_tr = eff_p_tr, l2_norm = l2_norm, l2_norm_sq = l2_norm_sq))
}


compute_metrics_from_xgboost = function(model,data,target){
  
  pred = predict(model, data)
  
  # Compute RMSE
  mse = mean((pred - target)^2)
  
  # compute accuracy
  acc = mean((pred > 0) == (target > 0))
  
  
  
  
  return(list(mse = mse,acc = acc))
  
  
  
}

reconstruct_predictions = function(bst, dtest) {
  # Get leaf indices from the model
  leaf_indices = predict(bst, dtest, predleaf = TRUE)
  
  # Extract booster model and tree data
  booster = xgb.Booster.complete(bst)
  trees = xgb.model.dt.tree(model = booster)
  
  # Initialize a matrix to store the predicted values for each sample
  predicted_values = matrix(0, nrow = nrow(leaf_indices), ncol = ncol(leaf_indices))
  reconstructed_predictions = numeric(nrow(leaf_indices))
  
  # Loop through each row (sample) in the leaf indices matrix
  for (i in 1:nrow(leaf_indices)) {
    sample_prediction = 0
    # Loop through each column (tree) in the leaf indices matrix
    for (j in 1:ncol(leaf_indices)) {
      # Get the leaf node index for the current sample and tree
      leaf_node = leaf_indices[i, j]
      
      # Find the corresponding row in the trees data.table
      leaf_row = trees[Tree == (j - 1) & Node == leaf_node]
      
      sample_prediction = sample_prediction + leaf_row$Quality
      
      # Extract the predicted value (Quality) and store it in the predicted_values matrix
      predicted_values[i, j] = leaf_row$Quality
    }
    
    reconstructed_predictions[i] = sample_prediction
  }
  
  return(reconstructed_predictions)
}
