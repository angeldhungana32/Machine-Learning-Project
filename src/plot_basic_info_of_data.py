'''
    @author - Angel Dhungana
'''
import matplotlib.pyplot as plt
import read_csv
import numpy as np


def plt_info(Y):
    '''
        Reads the dataset and plots graphs classifying negative and positive reviews
        Also, plots the graph of ratings
    '''
    dict_of_stars = {}
    for y_da in Y:
        if y_da in dict_of_stars:
            dict_of_stars[y_da] += 1
        else:
            dict_of_stars[y_da] = 1
    positive_reviews1, negative_reviews1 = get_pos_neg(dict_of_stars, False)
    positive_reviews2, negative_reviews2 = get_pos_neg(dict_of_stars, True)
    plt_pos_neg_no_3(positive_reviews1, negative_reviews1)
    plt_pos_neg_3(positive_reviews2, negative_reviews2)
    plt_stars([
        dict_of_stars["1.0"], dict_of_stars["2.0"], dict_of_stars["3.0"],
        dict_of_stars["4.0"], dict_of_stars["5.0"]
    ])


def get_pos_neg(dict_of_stars, is_three_positive):
    if is_three_positive == True:
        return dict_of_stars["4.0"] + dict_of_stars["3.0"] + dict_of_stars[
            "5.0"], dict_of_stars["1.0"] + dict_of_stars["2.0"]
    else:
        return dict_of_stars["4.0"] + dict_of_stars["5.0"], dict_of_stars[
            "1.0"] + dict_of_stars["2.0"] + dict_of_stars["3.0"]


def plt_pos_neg_no_3(pos, neg):
    '''
        Plots the Positive and Negative Reviews Counting Star Rating 2 as Negative
    '''
    label = ["positive", "negative"]
    index = np.arange(len(label))
    bar_list = plt.bar(index, [pos, neg], width=0.1, align='center')
    bar_list[0].set_color('darkolivegreen')
    bar_list[1].set_color('firebrick')
    plt.ylabel("Counts", fontsize=10)
    plt.xticks(index, label, fontsize=7, rotation=30)
    plt.title('Number of Positive and Negative Reviews')
    txt = "Star 3 considered as negative"
    plt.figtext(
        0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=10)
    plt.savefig('pos_neg_no_3.pdf', bbox_inches='tight')
    plt.close()


def plt_pos_neg_3(pos, neg):
    '''
        Plots the Positive and Negative Reviews Counting Star Rating 3 as Negative
    '''
    label = ["positive", "negative"]
    index = np.arange(len(label))
    bar_list = plt.bar(index, [pos, neg], width=0.1)
    bar_list[0].set_color('darkolivegreen')
    bar_list[1].set_color('firebrick')
    plt.ylabel("Counts", fontsize=10)
    plt.xticks(index, label, fontsize=7, rotation=30)
    plt.title('Number of Positive and Negative Reviews')
    txt = "Star 3 considered as positive"
    plt.figtext(
        0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=10)
    plt.savefig('pos_neg_3.pdf', bbox_inches='tight')
    plt.close()


def plt_stars(stars):
    '''
        Plots the star ratings 
        Bar graph
    '''
    label = ["1", "2", "3", "4", "5"]
    index = np.arange(len(label))
    bar_list = plt.bar(index, stars)
    colors = ['chocolate', 'firebrick', 'yellowgreen', 'teal', 'slategray']
    for i in range(len(bar_list)):
        bar_list[i].set_color(colors[i])
    plt.xlabel("Ratings", fontsize=10)
    plt.ylabel("Counts", fontsize=10)
    plt.xticks(index, label, fontsize=10, rotation=30)
    plt.title('Ratings Count')
    plt.savefig('stars.pdf', bbox_inches='tight')
    plt.close()