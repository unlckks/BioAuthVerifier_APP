//
//  Untitled.swift
//  BioAuthVerifier
//
//  Created by mingyun zhang on 12/9/24.
//

import Foundation

// MARK: - Models for API Responses

// Model for the prediction response from the server. The following code is fine-tuned using chatgpt
struct PredictionResponse: Codable {
    let prediction: String // The prediction result, e.g., "real" or "fake"
    let probability: Double // The probability associated with the prediction
}

// Model for the evaluation response from the server
struct EvaluationResponse: Codable {
    let precision: Double // Precision metric for model evaluation
    let recall: Double // Recall metric for model evaluation
    let accuracy: Double // Accuracy metric for model evaluation
    let confusionMatrix: [[Int]] // Confusion matrix as a 2D array
    let confusionMatrixImage: String // URL or base64 encoded image of the confusion matrix
}

// MARK: - API Manager for BioAuth Service
class BioAuthAPI {
    static let shared = BioAuthAPI() // Singleton instance of the API manager
    private let baseURL = "http://10.9.141.88:8000" // Base URL for the backend server

    // Function to upload a file and receive a prediction response
    func uploadAndPredict(fileURL: URL, completion: @escaping (Result<PredictionResponse, Error>) -> Void) {
        // Construct the URL for the prediction endpoint
        guard let url = URL(string: "\(baseURL)/upload_and_predict/") else {
            completion(.failure(NSError(domain: "Invalid URL", code: 0, userInfo: nil))) // Handle invalid URL error
            return
        }

        var request = URLRequest(url: url) // Create a URL request
        request.httpMethod = "POST" // Set HTTP method to POST

        let boundary = UUID().uuidString // Unique boundary string for the multipart form data
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type") // Set Content-Type header

        var data = Data() // Initialize data object to hold the multipart form data
        data.append("--\(boundary)\r\n".data(using: .utf8)!) // Start the multipart form data
        data.append("Content-Disposition: form-data; name=\"file\"; filename=\"\(fileURL.lastPathComponent)\"\r\n".data(using: .utf8)!) // Add file metadata

        // Determine MIME type dynamically based on file extension
        let mimeType: String
        if fileURL.pathExtension.lowercased() == "mp3" {
            mimeType = "audio/mpeg" // MIME type for MP3 files
            data.append("Content-Type: \(mimeType)\r\n\r\n".data(using: .utf8)!) // Add MIME type to the form data
        } else {
            // Return error if the file is not an MP3
            completion(.failure(NSError(domain: "Unsupported file type", code: 0, userInfo: [NSLocalizedDescriptionKey: "Only MP3 files are supported."])))
            return
        }

        do {
            // Read the file data from the fileURL
            let fileData = try Data(contentsOf: fileURL)
            data.append(fileData) // Append the file data to the form data
        } catch {
            completion(.failure(error)) // Handle file reading error
            return
        }

        data.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!) // End the multipart form data

        // Create and execute the upload task
        URLSession.shared.uploadTask(with: request, from: data) { data, response, error in
            if let error = error {
                completion(.failure(error)) // Handle network error
                return
            }

            // Check if the server response is valid and status code is 200
            guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
                completion(.failure(NSError(domain: "Invalid response", code: 0, userInfo: nil))) // Handle invalid response error
                return
            }

            // Ensure there is data in the response
            guard let data = data else {
                completion(.failure(NSError(domain: "No data", code: 0, userInfo: nil))) // Handle no data error
                return
            }

            do {
                // Decode the response data into the `PredictionResponse` model
                let response = try JSONDecoder().decode(PredictionResponse.self, from: data)
                completion(.success(response)) // Return the successful response
            } catch {
                completion(.failure(error)) // Handle decoding error
            }
        }.resume() // Start the upload task
    }
}
