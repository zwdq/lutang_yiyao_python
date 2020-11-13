import pymysql.cursors

# 连接数据库
connect = pymysql.Connect(
    host='localhost',
    port=3306,
    user='root',
    passwd='123',
    db='mydb',
    charset='utf8'
)

# 获取游标
cursor = connect.cursor()

# 查询数据
sql = "SELECT name,money FROM mytable "
#data = ('13512345678',)
cursor.execute(sql)
print(cursor.fetchall())

# 插入数据
sql = "INSERT INTO mytable (name, money) VALUES ( '%s', '%s')"
data = ('雷军', '1000w')
cursor.execute(sql % data)
connect.commit()
print('成功插入', cursor.rowcount, '条数据')

# 关闭连接
cursor.close()
connect.close()