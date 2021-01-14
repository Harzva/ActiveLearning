import os, sys

retval = os.getcwd()
print ("当前工作目录为 %s" % retval)

# 修改当前工作目录
os.chdir( 'UFSLviaIC' )

# 查看修改后的工作目录
retval = os.getcwd()

print ("目录修改成功 %s" % retval)