def get_indexes(sentence):
    start = "<s>"
    end = "<e>"
    my_list = sentence.split()
    last = ""
    indexes_start = []
    indexes_end = []
    missed = 0
    for i, element in enumerate(my_list):
        if element == start:
            if last == start:
                indexes_end.append(-1)
                missed += 1
            indexes_start.append(i - len(indexes_start) - len(indexes_end) + missed)
            last = start
        elif element == end:
            if last == end:
                indexes_start.append(-1)
                missed += 1
            indexes_end.append(i - len(indexes_start) - len(indexes_end) + missed)
            last = end

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
    grade = grade / length
    return grade

def get_qualtiy(excepted, results):
    excepted = "a b c d e f g h i j k l"
    results = "a b <s> c d e f g h i j k l"
    scores = []
    excepted_indexes_start, excepted_indexes_end = get_indexes(excepted)
    results_indexes_start, results_indexes_end = get_indexes(results)

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

    score = 0
    for s in scores:
        score += s
    if len(scores) > 0:
        score /= len(scores)       
    score = 1 - score
    print(score)

get_qualtiy("","")
