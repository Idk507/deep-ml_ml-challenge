def batch_requests(requests: list, max_batch_size: int, max_wait_time: float) -> list:
    """
    Group inference requests into batches based on size and time constraints.
    
    Args:
        requests: List of dicts with 'id', 'timestamp', 'features'
        max_batch_size: Maximum number of requests per batch
        max_wait_time: Maximum time to wait before processing a batch
    
    Returns:
        List of tuples: (request_ids, batched_features, process_time)
    """
    # Handle empty input
    if not requests:
        return []
    
    # Sort requests by timestamp to process in chronological order
    sorted_requests = sorted(requests, key=lambda r: r['timestamp'])
    
    batches = []
    current_batch_ids = []
    current_batch_features = []
    batch_start_time = None
    
    for request in sorted_requests:
        req_id = request['id']
        timestamp = request['timestamp']
        features = request['features']
        
        # If this is the first request in a new batch
        if batch_start_time is None:
            batch_start_time = timestamp
            current_batch_ids.append(req_id)
            current_batch_features.append(features)
        else:
            # Check if adding this request would violate constraints
            wait_time = timestamp - batch_start_time
            would_exceed_size = len(current_batch_ids) >= max_batch_size
            would_exceed_time = wait_time > max_wait_time
            
            # If either constraint would be violated, process current batch first
            if would_exceed_size or would_exceed_time:
                # Get the processing timestamp (last request's timestamp in current batch)
                process_time = sorted_requests[sorted_requests.index(request) - 1]['timestamp']
                batches.append((
                    current_batch_ids,
                    current_batch_features,
                    round(process_time, 4)
                ))
                
                # Start new batch with current request
                current_batch_ids = [req_id]
                current_batch_features = [features]
                batch_start_time = timestamp
            else:
                # Add request to current batch
                current_batch_ids.append(req_id)
                current_batch_features.append(features)
    
    # Process the last batch if it has any requests
    if current_batch_ids:
        # The processing time is the timestamp of the last request in the batch
        process_time = sorted_requests[-1]['timestamp']
        batches.append((
            current_batch_ids,
            current_batch_features,
            round(process_time, 4)
        ))
    
    return batches
