docker build -t twilio-cr-gemini .

docker run -p 8520:8520 \
  -e NGROK_URL="https://your-ngrok-url" \
  -e TWILIO_ACCOUNT_SID="your_sid" \
  -e TWILIO_AUTH_TOKEN="your_token" \
  -e TWILIO_PHONE_NUMBER="your_number" \
  -e GOOGLE_API_KEY="your_key" \
  twilio-cr-gemini


  docker run -p 8520:8520 --env-file .env twilio-cr-gemini