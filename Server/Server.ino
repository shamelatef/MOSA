#include <ESP8266WiFi.h>
int LED = 5; // Assign LED pin i.e: D1 on NodeMCU
const int trigPin = 12;
const int echoPin = 14;

//define sound velocity in cm/uS
#define SOUND_VELOCITY 0.034
#define CM_TO_INCH 0.393701

long duration;
float distanceCm;
float distanceInch;

WiFiServer server(80); // Create a server object
WiFiClient client;




void setup() {
  pinMode(LED, OUTPUT);
  Serial.begin(115200); // Start the serial communication
  WiFi.begin("OMAR", "12345678");
  Serial.begin(115200); // Starts the serial communication
  pinMode(trigPin, OUTPUT); // Sets the trigPin as an Output
  pinMode(echoPin, INPUT); // Sets the echoPin as an Input
 // Connect to your WiFi network
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi");
  server.begin(); // Start the server
}

void loop() {
  WiFiClient client = server.available();
  Serial.println(WiFi.localIP());
   //WiFiClient client2 = server.available(); // Check for incoming clients
  if (client) {
    Serial.println();
    Serial.println("New client connected.");
    while (client.connected()) {
      if (client.available()) {
        char signal = client.read(); // Read the signal from the client
        if (signal == '1') {
            // Clears the trigPin
            digitalWrite(trigPin, LOW);
            delayMicroseconds(2);
            // Sets the trigPin on HIGH state for 10 micro seconds
            digitalWrite(trigPin, HIGH);
            delayMicroseconds(10);
            digitalWrite(trigPin, LOW);
            
            // Reads the echoPin, returns the sound wave travel time in microseconds
            duration = pulseIn(echoPin, HIGH);
            
            // Calculate the distance
            distanceCm = duration * SOUND_VELOCITY/2;
            
            // Convert to inches
            distanceInch = distanceCm * CM_TO_INCH;
            
            // Prints the distance on the Serial Monitor
            Serial.print("Distance (cm): ");
            Serial.println(distanceCm);
            
            // Send a signal to the other NodeMCU
            if (distanceCm < 100) 
            {
              Serial.print("heheheh ");
              digitalWrite(LED, HIGH); // turn the LED on
              delay(5000); // wait for a second
              digitalWrite(LED, LOW); // turn the LED off
            }

            
            delay(1000);
  
          }
      }
    }
    client.stop(); // Stop the client
    Serial.println("Client disconnected.");
  }
}
