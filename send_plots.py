import matplotlib.pyplot as plt
from IPython import display as diss

plt.ion()

def plotting(score, mean_score):
    diss.clear_output(wait=True)
    diss.display(plt.gcf())
    plt.clf()
    plt.title('MOMMM!!!! CANT YOU SEE IM GAMING HERE!!!!!')
    plt.xlabel('Games being Gamed')
    plt.ylabel('POGCHAMPS')
    # plt.plot(score)
    plt.plot(mean_score)
    plt.ylim(ymin=0)
    plt.text(len(score)-1, score[-1], str(score[-1]))
    plt.text(len(mean_score)-1, mean_score[-1], str(mean_score[-1]))
    plt.show(block=False)
    plt.pause(.1)