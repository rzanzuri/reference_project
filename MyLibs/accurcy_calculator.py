import numpy as np

def levenshtein_edit_dis(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    # print (matrix)
    return (matrix[size_x - 1, size_y - 1])

def get_indexes(sentence, start_tag, end_tag):
  my_list = sentence.split()
  last = ""
  indexes_start = []
  indexes_end = []
  missed = 0
  for i, element in enumerate(my_list):
      if element == start_tag:
          if last == start_tag:
              indexes_end.append(-1)
              missed += 1
          indexes_start.append(i - len(indexes_start) - len(indexes_end) + missed)
          last = start_tag
      elif element == end_tag:
          if last == end_tag:
              indexes_start.append(-1)
              missed += 1
          indexes_end.append(i - len(indexes_start) - len(indexes_end) + missed)
          last = end_tag

  if len(indexes_start) > len(indexes_end):
      for i in range(len(indexes_start) - len(indexes_end)):
          indexes_end.append(-1)

  if len(indexes_end) > len(indexes_start):
      for i in range(len(indexes_end) - len(indexes_start)):
          indexes_start.append(-1)

  return indexes_start, indexes_end

def get_grade(index_a, index_b, length):
  if index_a == -1 or index_b == -1: return 1
  grade = abs(index_a - index_b)
  grade = grade / (length / 2 )
  return grade

def get_accurcy(excepted, results, start_tag, end_tag, special_tags = [] ,accuracy = 0.5):
    similarity = get_sentences_smilarity(excepted, results, special_tags ,accuracy)
    ref_accurcy = get_reference_accurcy(excepted, results, start_tag, end_tag, special_tags,accuracy)
    if ref_accurcy == -1: ref_accurcy = 1
    return  (similarity + ref_accurcy) / 2

def get_reference_accurcy(excepted, results, start_tag, end_tag, special_tags = [] ,accuracy = 0.5):        
    scores = []
    similarity = get_sentences_smilarity(excepted, results, special_tags, accuracy)
    excepted_indexes_start, excepted_indexes_end = get_indexes(excepted, start_tag, end_tag)
    results_indexes_start, results_indexes_end = get_indexes(results, start_tag, end_tag)

    if not does_check_reference(excepted, start_tag, end_tag, similarity, accuracy):
        return -1

    while len(results_indexes_start) > len(excepted_indexes_start) and -1 in results_indexes_start:
        results_indexes_start.remove(-1)
    while len(results_indexes_end) > len(excepted_indexes_end) and -1 in results_indexes_end:
        results_indexes_end.remove(-1)
    for index in excepted_indexes_start[:]:
        if index in results_indexes_start:
            excepted_indexes_start.remove(index)
            results_indexes_start.remove(index)
            scores.append(0)
    for index in excepted_indexes_end[:]:
        if index in results_indexes_end:
            excepted_indexes_end.remove(index)
            results_indexes_end.remove(index)
            scores.append(0)
    for i in range(max(len(excepted_indexes_start), len(results_indexes_start))):
        if len(excepted_indexes_start) > 0 and len(results_indexes_start) > 0:
            scores.append(get_grade(excepted_indexes_start.pop(), results_indexes_start.pop() , len(excepted.split())))
        else:
            scores.append(1)

    for i in range(max(len(excepted_indexes_end), len(results_indexes_end))):
        if len(excepted_indexes_end) > 0 and len(results_indexes_end) > 0:
            scores.append(get_grade(excepted_indexes_end.pop(), results_indexes_end.pop() , len(excepted.split())))
        else:
            scores.append(1)

    score = 0.0
    for s in scores:
        score += s
    if  len(scores) > 0: score /= len(scores)
    
    score = 1 - score
    return score

def get_sentences_smilarity(excepted, actual, special_tags = [] ,accuracy = 0.5):
    for tag in special_tags:        
        excepted = excepted.replace(tag,'')
        actual = actual.replace(tag,'')
    total_words = excepted.split()
    num_of_mistaks = levenshtein_edit_dis(excepted.split(), actual.split())

    similarity = 1 - (num_of_mistaks / len(total_words))
   
    return similarity if similarity >= accuracy else 0.0

def does_check_reference(sentence, start_tag, end_tag, similarity ,accuracy):
    return start_tag in sentence and end_tag in sentence and similarity >= accuracy

def get_total_accurcy(accurcy_list):
    if len(accurcy_list) > 0:
        return (sum(accurcy_list) / len(accurcy_list))
    return 0.0

def get_total_ref_accurcy(accurcy_list):
    total_acc = 0
    sum_acc = 0

    for acc in accurcy_list:
        if acc > -1:
            total_acc += 1
            sum_acc += acc

    if total_acc == 0:
        return 0.0           
    return (sum_acc / total_acc)