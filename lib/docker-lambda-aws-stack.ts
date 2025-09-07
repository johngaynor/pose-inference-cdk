import * as cdk from "aws-cdk-lib";
import { Construct } from "constructs";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as iam from "aws-cdk-lib/aws-iam";

export class DockerLambdaAwsStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // 1️⃣ Create the IAM role for Lambda
    const s3LambdaRole = new iam.Role(this, "S3LambdaRole", {
      assumedBy: new iam.ServicePrincipal("lambda.amazonaws.com"),
      description: "Role for Lambda to access S3",
    });

    // Attach policies to allow S3 access
    s3LambdaRole.addManagedPolicy(
      iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonS3FullAccess")
    );

    // Attach basic Lambda execution role for CloudWatch logging
    s3LambdaRole.addManagedPolicy(
      iam.ManagedPolicy.fromAwsManagedPolicyName(
        "service-role/AWSLambdaBasicExecutionRole"
      )
    );

    // 2️⃣ Define the Docker Lambda and assign the role
    const dockerFunction = new lambda.DockerImageFunction(
      this,
      "DockerFunction",
      {
        functionName: "pose-inference",
        code: lambda.DockerImageCode.fromImageAsset("./image"),
        memorySize: 1024,
        timeout: cdk.Duration.seconds(120),
        role: s3LambdaRole,
      }
    );
  }
}
