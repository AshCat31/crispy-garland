import os.path


def pre_mutation(context):
    dirname, filename = os.path.split(context.filename)
    testfile = "test_" + filename
    print(context.config.test_command)
    context.config.test_command += ' ' + os.path.join('tests', dirname, testfile)
    # print(dirname, testfile)
    # print(context.config.test_command)
