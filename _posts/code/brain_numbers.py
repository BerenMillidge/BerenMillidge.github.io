# visualizing and plotting the data from blue brain cell atlas for mouse
# to understand how mouse brain etc relates to AI models
import numpy as np
import matplotlib.pyplot as plt


FORMAT = "png"
# this is all for the MOUSE
def brain_regions_pie():
    total_neurons = 105653652
    # of which excitatory 57 162 675 and inhibitory = 14 534 967

    cerebal_cortex = 38913718
    hippocampus = 7333388
    basal_ganglia = 5221071
    thalamus = 2958946
    hypothalamus = 2374136
    midbrain = 4413483
    cerebellum = 48097875
    hindbrain = 3674423

    sizes = [cerebal_cortex,thalamus,basal_ganglia,hypothalamus,hippocampus,midbrain,hindbrain,cerebellum]

    labels = ['Cortex','Thalamus','Basal Ganglia','Hypothalamus','Hippocampus','Midbrain','Hindbrain','Cerebellum']

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.0f%%',
            shadow=False, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.set_title("Fraction of neurons in main brain structures (mouse)", fontsize=14, pad=20)
    plt.savefig("fractions_main_brain_regions_mouse." + FORMAT, format=FORMAT)
    plt.show()

def cortical_regions_pie():
    olfactory = 10402662
    somatosensory = 2927777
    somatomotor = 2927777
    gustatory  = 242804
    auditory = 1622262
    visual = 2565173
    anterior_cingulate = 770227
    posterior_parietal = 683860
    temporal = 682203
    perirhinal = 206843
    entorhinal = 64025
    agranular_insula =508275
    orbital = 599296
    frontal_pole = 28248
    prelimbic = 261181
    infralimbic_area = 283406
    retrosplenial = 1662829

    # made up combinations
    association_areas = posterior_parietal + temporal
    hippocampus_adjacent = perirhinal + entorhinal + retrosplenial
    frontal = orbital + frontal_pole
    limbic = anterior_cingulate + agranular_insula + prelimbic + infralimbic_area

    sizes = [olfactory, somatosensory, somatomotor, auditory, visual, association_areas, frontal, hippocampus_adjacent, limbic]
    labels = ["Olfaction", "Somatosensory","Motor","Audition","Vision", "Association areas","Frontal cortex","Hippocampal system","Limbic system"]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.0f%%',
            shadow=False, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.set_title("Fraction of neurons in mouse cortical areas", fontsize=14,pad=20)
    plt.savefig("mouse_cortex_fractions." + FORMAT, format=FORMAT)
    plt.show()


def all_cortical_regions_pie():
    olfactory = 10402662
    somatosensory = 2927777
    somatomotor = 2927777
    gustatory  = 242804
    auditory = 1622262
    visual = 2565173
    anterior_cingulate = 770227
    posterior_parietal = 683860
    temporal = 682203
    perirhinal = 206843
    entorhinal = 64025
    agranular_insula =508275
    orbital = 599296
    frontal_pole = 28248
    prelimbic = 261181
    infralimbic_area = 283406
    retrosplenial = 1662829

    sizes = [olfactory, somatosensory, somatomotor, auditory, visual, gustatory,anterior_cingulate,posterior_parietal,temporal,perirhinal, agranular_insula,orbital,prelimbic, infralimbic_area,retrosplenial]
    
    labels = ["Olfaction", "Somatosensory","Motor","Audition","Vision", "Gustatory","Anterior Cingulate","Posterior Parietal","Temporal","Perirhinal","Insula","Orbital","Prelimbic","Infralimbic","Retrosplenial"]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.0f%%',
            shadow=False, startangle=90,textprops = {"fontsize": 10})
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.set_title("Fraction of neurons in mouse cortical areas", fontsize=14,pad=20)
    plt.savefig("all_mouse_cortex_fractions." + FORMAT, format=FORMAT)
    plt.show()



if __name__ == '__main__':
    #brain_regions_pie()
    #cortical_regions_pie()
    all_cortical_regions_pie()

