import glob
import numpy as np
# Project3


def readStopWord(filename):
    stopword = []
    f = open(filename, "r")
    for i in f.readlines():
        stopword.append(i.strip('\n').strip(' '))
    return stopword


filename = 'stopwords'
stop_words = readStopWord(filename)


# groups包含20个列表，每个列表是每一类所有文本的名字，大约1000个元素,这个方法得到所有词典
def getdictionary(groups):
    alldictionary = []
    for group in groups:
        for i in range(0, int((len(group)+1)/2)):
            with open(group[i], 'r') as file:
                lines = file.readlines()
                for line in lines:
                    line = line.strip()
                    i = 0
                    while i < len(line):
                        word = ""
                        if not valid(line[i]):
                            i += 1
                            continue
                        while i < len(line) and valid(line[i]):
                            if 'A' <= line[i] <= 'Z':
                                word += line[i].lower()
                            else:
                                word += line[i]
                            i += 1
                        if word in stop_words or len(word) is 1:
                            continue
                        if word not in alldictionary:
                            alldictionary.append(word)
    print("complete getdictionary, go to next step...")
    return alldictionary


def getclassdict(groups, filenum, flag):
    alldictionary = getdictionary(groups)
    # traintxtnum是训练集文本的个数，约等于500*20(-2). len(alldictionary)是数组的列数，代表词典中的总词数
    class_dictionary = np.zeros((filenum, len(alldictionary)))
    group_dictionary = np.zeros((20, len(alldictionary)))
    row = 0
    linewordsnum = []
    groupwordsum = []
    groupindex = 0
    for group in groups:
        count = 0
        if flag is 0:
            start = 0
            end = int((len(group) + 1) / 2)
        else:
            start = int((len(group) + 1) / 2)
            end = len(group)
        for i in range(start, end):
            linecount = 0
            with open(group[i], 'r') as file:
                lines = file.readlines()
                for line in lines:
                    line = line.strip()
                    i = 0
                    while i < len(line):
                        word = ""
                        if not valid(line[i]):
                            i += 1
                            continue
                        while i < len(line) and valid(line[i]):
                            if 'A' <= line[i] <= 'Z':
                                word += line[i].lower()
                            else:
                                word += line[i]
                            i += 1
                        if word in stop_words or len(word) is 1 or word not in alldictionary:
                            continue
                        class_dictionary[row][alldictionary.index(word)] += 1
                        group_dictionary[groupindex][alldictionary.index(word)] += 1
                        count += 1
                        linecount += 1
            linewordsnum.append(linecount)
            row += 1
        groupwordsum.append(count)
        groupindex += 1
    print("complete getclassdict, go to next step...")
    return class_dictionary, groupwordsum, linewordsnum, group_dictionary


def gettfidf(class_dictionary, traingroupwords, groups, traingroup_dictionary):
    class_dict = np.mat(class_dictionary)
    tf = np.zeros((len(traingroup_dictionary), class_dict.shape[1]))
    idf = np.zeros((len(traingroup_dictionary), class_dict.shape[1]))
    traingroup_dictionary = np.array(traingroup_dictionary)
    column = np.zeros((1, class_dict.shape[1]))
    for col in range(class_dict.shape[1]):
        num = 0
        for row in range(len(class_dict)):
            if class_dict[row, col] > 0:
                num += 1
        num += 1
        column[0, col] = num
    loop = 0
    while loop < len(traingroup_dictionary):
        tf[loop, :] = traingroup_dictionary[loop, :]/traingroupwords[loop]
        idf[loop, :] = np.log10(len(class_dict)/column[0, :])
        loop += 1
    print("tf:", tf)
    print("idf:", idf[0])
    tfidf = np.zeros((len(traingroup_dictionary), class_dict.shape[1]))
    for m in range(len(tfidf)):
        for n in range(tfidf.shape[1]):
            tfidf[m, n] = tf[m, n] * idf[m, n]
    traingrouplen = []
    testgrouplen = []
    for group in range(len(groups)):
        if group is 0:
            traingrouplen.append(len(groups[0]) / 2)
            testgrouplen.append(len(groups[0]) - len(groups[0]) / 2)
        else:
            traingrouplen.append(traingrouplen[group-1]+int((len(groups[group])+1)/2))
            testgrouplen.append(traingrouplen[group-1]+len(groups[group])-int((len(groups[group])+1)/2))
    print('tfidf:', tfidf)
    print('complete gettfidf, go to next step...')
    return tfidf, traingrouplen, testgrouplen


# train_dictionary：测试集文件的总数作为行数，所有词数作为列数    trainlinewords：每行的单词总数
# tfidf: tfidf表     traingrouplen：列表，保存20个类各自训练集的数目        testgrouplen： 列表，保存20个类各自测试集的数目
def nb(test_dictionary, tfidf, testgrouplen, testlenwords):
    correct = 0
    for testrow in range(len(test_dictionary)):      #testrow代表是第几个test文件
        print("Processing the", testrow,  "th test text(total 9998)...")
        row = 0
        maxrow = 1
        maxpb = (-2147483647 - 1)
        while row < len(tfidf):   # 20行， 每行代表每一类训练集
            pb = 0
            for testcol in range(tfidf.shape[1]):
                pb += np.log((tfidf[row, testcol]) + 1)*test_dictionary[testrow, testcol]
            if pb > maxpb:
                maxpb = pb
                maxrow = row     # maxrow预测属于第几类
            row += 1
        # 判断testrow属于第几类
        realcls = 0
        for i in range(len(testgrouplen)):
            if testrow < testgrouplen[i]:
                realcls = i
                break
            elif testrow is testgrouplen[i]:
                realcls = i+1
                break
        if realcls == maxrow:
            correct += 1
    print("The accuracy is: ", correct/9998)
    print("Done!")
    return correct/9998


def valid(c):
    if 'a' <= c <= 'z' or 'A' <= c <= 'Z' or '0' <= c <= '9':
        return True
    else:
        return False


if __name__ == "__main__":
    groups = []
    totalfile = 0
    groups.append(glob.glob(r'20_newsgroups/alt.atheism/*'))
    totalfile += len(groups[len(groups)-1])
    groups.append(glob.glob(r'20_newsgroups/comp.graphics/*'))
    totalfile += len(groups[len(groups) - 1])
    groups.append(glob.glob(r'20_newsgroups/comp.os.ms-windows.misc/*'))
    totalfile += len(groups[len(groups) - 1])
    groups.append(glob.glob(r'20_newsgroups/comp.sys.ibm.pc.hardware/*'))
    totalfile += len(groups[len(groups) - 1])
    groups.append(glob.glob(r'20_newsgroups/comp.sys.mac.hardware/*'))
    totalfile += len(groups[len(groups) - 1])
    groups.append(glob.glob(r'20_newsgroups/comp.windows.x/*'))
    totalfile += len(groups[len(groups) - 1])
    groups.append(glob.glob(r'20_newsgroups/misc.forsale/*'))
    totalfile += len(groups[len(groups) - 1])
    groups.append(glob.glob(r'20_newsgroups/rec.autos/*'))
    totalfile += len(groups[len(groups) - 1])
    groups.append(glob.glob(r'20_newsgroups/rec.motorcycles/*'))
    totalfile += len(groups[len(groups) - 1])
    groups.append(glob.glob(r'20_newsgroups/rec.sport.baseball/*'))
    totalfile += len(groups[len(groups) - 1])
    groups.append(glob.glob(r'20_newsgroups/rec.sport.hockey/*'))
    totalfile += len(groups[len(groups) - 1])
    groups.append(glob.glob(r'20_newsgroups/sci.crypt/*'))
    totalfile += len(groups[len(groups) - 1])
    groups.append(glob.glob(r'20_newsgroups/sci.electronics/*'))
    totalfile += len(groups[len(groups) - 1])
    groups.append(glob.glob(r'20_newsgroups/sci.med/*'))
    totalfile += len(groups[len(groups) - 1])
    groups.append(glob.glob(r'20_newsgroups/sci.space/*'))
    totalfile += len(groups[len(groups) - 1])
    groups.append(glob.glob(r'20_newsgroups/soc.religion.christian/*'))
    totalfile += len(groups[len(groups) - 1])
    groups.append(glob.glob(r'20_newsgroups/talk.politics.guns/*'))
    totalfile += len(groups[len(groups) - 1])
    groups.append(glob.glob(r'20_newsgroups/talk.politics.mideast/*'))
    totalfile += len(groups[len(groups) - 1])
    groups.append(glob.glob(r'20_newsgroups/talk.politics.misc/*'))
    totalfile += len(groups[len(groups) - 1])
    groups.append(glob.glob(r'20_newsgroups/talk.religion.misc/*'))
    totalfile += len(groups[len(groups) - 1])
    trainfilenum = int((totalfile+1)/2)
    testfilenum = totalfile - trainfilenum

    test_dictionary, testgroupwords, testlinewords, testgroup_dictionary = getclassdict(groups, testfilenum, 1)
    train_dictionary, traingroupwords, trainlinewords, traingroup_dictionary = getclassdict(groups, trainfilenum, 0)

    tfidf, traingrouplen, testgrouplen = gettfidf(train_dictionary, traingroupwords, groups, traingroup_dictionary)

    nb(test_dictionary, tfidf, testgrouplen, testlinewords)
