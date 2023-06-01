# Build and deploy

Command to build the application. PLease remeber to change the project name and application name
```
gcloud builds submit gcr.io/bike-share-388211/washington-dc-bikeshare-app --project=bike-share-388211
```

Command to deploy the application
```
gcloud run deploy --image gcr.io/bike-share-388211/washington-dc-bikeshare-app --platform managed  --project=bike-share-388211 --allow-unauthenticated
```