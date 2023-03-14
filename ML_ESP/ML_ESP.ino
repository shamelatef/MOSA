#include <ESP8266WiFi.h>

const int ledPin = 5; // Pin connected to the LED
WiFiServer server(80); // Create a server object

void setup() {
  pinMode(ledPin, OUTPUT); // Set the LED pin as output
  Serial.begin(115200); // Start the serial communication
  WiFi.begin("TEdataDD2253", "12931530"); // Connect to your WiFi network
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi");
  server.begin(); // Start the server
}

void loop() {
      //Serial.println(WiFi.localIP()); // Print the local IP address

  WiFiClient client = server.available(); // Check for incoming clients
  if (client) {
    Serial.println();
    Serial.println("New client connected.");
    while (client.connected()) {
      if (client.available()) {
        char signal = client.read(); // Read the signal from the client
        if (signal == '1') {
          Serial.println("one");
          digitalWrite(ledPin, HIGH); // Turn on the LED
        }
        else if (signal == '0') {
          Serial.println("Zero");
          digitalWrite(ledPin, LOW); // Turn off the LED
        }
      }
    }
    client.stop(); // Stop the client
    Serial.println("Client disconnected.");
  }
}
