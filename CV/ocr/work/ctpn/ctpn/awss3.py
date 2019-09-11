import boto3

#s3 = boto3.client('s3')

s3 = boto3.client(service_name='s3',
        region_name='us-east-1',
        aws_access_key_id='your_ak',
        aws_secret_access_key='your_sk',
        config=None)

bucket_name = 'your_bucket'
inputFolder = "ctpn"
outputFolder = inputFolder+"/results"

def getInputImage(filename):
        s3.download_file( bucket_name, inputFolder+'/'+filename, './data/demo/'+filename)
        
def uploadResult(filename):
        s3.upload_file('./data/results/'+filename, bucket_name, outputFolder+'/'+filename)

if __name__ == '__main__':
        #getInputImage('1.png')
        uploadResult('004.jpg')
