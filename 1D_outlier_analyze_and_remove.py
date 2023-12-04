import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t
import math

# Box plot can be used for outlier detection
center_list= [4.724875033139281, 4.522433686469234, 4.575947735383538, 4.732236999001375, 4.70431725403129, 4.7216701610998015, 4.4706446115583045, 4.525577952360017, 4.7065565680623545, 4.722115317510147, 5.007679421931495, 4.716649073029788, 4.590594050309988, 4.6649227380331215, 4.711874641972846, 4.431008396904731, 4.7129776623946995, 4.711874700345085, 4.57596829835072, 4.659151607863959, 4.711029308348625, 4.685606393831816, 4.516971846974534, 4.649309876455512, 4.753013998509602, 4.710356378428987, 4.710257888797023, 4.4162281490039765, 4.510401421623406, 4.677738791427306, 4.718909821059418, 4.7067518863527065, 4.686890982262614, 4.633535530127102, 4.705009311031577, 4.62612641793356, 4.701816155231398, 4.7081302778773075, 4.51508098829686, 4.677940943622754, 4.706601052656165, 4.718784200345589, 4.729237468456142, 4.476142320157972, 4.705672002091032, 4.585035658333842, 4.637488443129749, 4.708821104686931]  # Your data
center_list = np.array(center_list)
sns.boxenplot(x=center_list)  
plt.show()

# Data set obtained using the z-test. For large data sets
def remove_outliers(data):
    n=len(data)
    mean = np.mean(data)
    std_dev = np.std(data)
    threshold = t.ppf(0.90, df=2000)
    z_scores = [(x - mean) / (std_dev/math.sqrt(n)) for x in data]
    filtered_data = [data[i] for i in range(len(data)) if abs(z_scores[i]) <= threshold]
    return filtered_data

print(remove_outliers(center_list))

# Using the t-test to identify outliers
data = center_list # Your data
def t_test(data):
    data = data.flatten()
    n = len(data)
    s = np.std(data, ddof=1)
    t_table = t.ppf(0.90, df=n-1)
    clean_data = []
    for i in range(1, n-1):
        q_calculated = (data[i] - data[i-1]) / (s/math.sqrt(n))
        if q_calculated > t_table:
            print(f"Outlier detected: {data[i]}")
            clean_data.append(data[i])
    return np.array(clean_data)

# Call the Q-test
t_test(data)