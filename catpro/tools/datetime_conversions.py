import pandas as pd
import numpy as np
import datetime
import pytz
from dateutil import parser


def utc0_to_timezone(datetime_value, timezone = 'America/New_York',output_format = "%Y-%m-%d %H:%M:%S"):
	'''

	Args:
		datetime_value:
		timezone:
		output_format:

	Returns:

	# Example
	timezone = 'America/New_York'
	output_format = "%Y-%m-%d %H:%M:%S"
	datetime_value =np.datetime64('2020-01-17T22:50:00.000000000')
	datetime_value = pd.Timestamp(datetime_value)
	datetime_value = datetime_value.replace(tzinfo=pytz.utc)
	datetime_value = datetime_value.astimezone(pytz.timezone(timezone ))
	datetime_str = datetime_value.strftime(output_format)
	'''

	if pd.isnull(datetime_value):
		# print("Warning: null datetime. returning np.datetime64('NaT'), np.nan")
		return np.datetime64('NaT'), np.nan
	datetime_value = pd.Timestamp(datetime_value)
	datetime_value = datetime_value.replace(tzinfo=pytz.utc) #put in UTC-0
	datetime_value = datetime_value.astimezone(pytz.timezone(timezone ))
	datetime_str = datetime_value.strftime(output_format)
	return datetime_value, datetime_str


def add_timezone(datetime_value, timezone='America/New_York'):
	tz = pytz.timezone(timezone)
	datetime_value = tz.localize(datetime_value)
	return datetime_value


# from dateutil import parser
from datetime import datetime
def str_time_to_datetime(str_time='2020-12-22 14:02:01', method='manual', input_format = '%Y-%m-%d %H:%M:%S', add_timezone = 'America/New_York'):
	# timezones: https://gist.github.com/heyalexej/8bf688fd67d7199be4a1682b3eec7568
	if method == 'automated':
		# automated but slower and not sure about performance
		# ======================================================
		dt = parser.parse(str_time)
	elif method == 'manual':
		# https://strftime.org/
		dt = datetime.strptime(str_time, input_format)
	if add_timezone:
		tz = pytz.timezone(add_timezone)
		dt = tz.localize(dt)

	return dt

def datetime_to_str_time(datetime_value,output_format = "%Y-%m-%d %H:%M:%S"):
	return datetime_value.strftime(output_format)


def nearest(list_of_datetimes, target_datetime):
	return min(list_of_datetimes, key=lambda x: abs(x - target_datetime))

	# MANUAL v2
	# ========================================================================
	# if str(str_time)=='nan':
	#     return np.nan
	# date = str_time.split(' ')[0].split('-')
	# time = str_time.split(' ')[1].split(':')
	# t = [int(n) for n in np.concatenate([date, time])]
	# dt = datetime.datetime(t[0], t[1], t[2], t[3], t[4], t[5])
	# return dt
'''
print(str_time_to_datetime(str_time='2020-12-22 14:02:01', method='manual'))
'''