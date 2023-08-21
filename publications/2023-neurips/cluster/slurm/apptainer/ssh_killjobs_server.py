import json
import pymysql
import pandas as pd
import time

pw2 = 'database_password'

def postprocess_table(table_name):

	cnx = pymysql.connect(host='lcdb_experiments.ewi.tudelft.nl', user='lcdb', passwd=pw2, db='db_lcdb')
	query = '''select * from %s where postprocess=1;''' % table_name
	to_process = pd.read_sql_query(query, cnx)

	print('found %d rows for processing...' % len(to_process))

	query_list = []

	for i in range(0, len(to_process)):
		print('working on row %d' % i)

		row = to_process.iloc[i]

		query = '''select * from %s where workflow='%s' and openmlid=%d and hyperparameters='%s' and status='created';''' % (
		table_name, row.workflow, row.openmlid, row.hyperparameters)

		datas = pd.read_sql_query(query, cnx)
		if len(datas) < 1:
			print('this row doesnt have any jobs remaining... too bad!')
		else:
			trainsize_small = json.loads(row.train_sizes)[0]

			trainsizes_todo = []
			for train_size in datas['train_sizes'].unique():
				train_size_ = json.loads(train_size)
				if train_size_[0] > trainsize_small:
					trainsizes_todo.append(train_size)

			for trainsize in trainsizes_todo:
				query_list.append(
					'''update %s set status='skipped' where workflow='%s' and openmlid=%d and hyperparameters='%s' and status='created' and train_sizes='%s';''' % (
					table_name, row.workflow, row.openmlid, row.hyperparameters, trainsize))

		query_list.append('''update %s set postprocess=0 where id=%d''' % (table_name, row.ID))

	print('I have to execute %d queries... Lets get to work!' % len(query_list))

	affected_rows = []
	if len(query_list) > 0:
		cursor = cnx.cursor()
		for query in query_list:
			print('performing query: %s' % query)
			tmp = (cursor.execute(query))
			print('rows affected: %d' % tmp)
			affected_rows.append(tmp)
		cursor.close()
		cnx.commit()
	cnx.close()


while True:
	try:
		print('trying small...')
		postprocess_table('jobs_small')
		print('trying medium...')
		postprocess_table('jobs_medium')
		print('trying large...')
		postprocess_table('jobs_large')
	except Exception as e:
		print('failed with error %s' % str(e))
	print('going to sleep for 5 min...')
	time.sleep(60*5)