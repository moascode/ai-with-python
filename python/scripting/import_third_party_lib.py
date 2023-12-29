#pip install pytz #install specific library
#pip install -r requirements.txt # install all required libraries
    #pytz==2016.7
    #requests==2.11.1
import pytz
from datetime import datetime

utc = pytz.utc
kl = pytz.timezone('Asia/Kuala_Lumpur')

now = datetime.now(utc)
kl_now = datetime.now(kl)

print(now)
print(kl_now)