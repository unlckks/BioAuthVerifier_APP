//
//  ViewController.swift
//  BioAuthVerifier
//
//  Created by mingyun zhang on 12/9/24.
//
//

import UIKit
import AVFoundation
// The following code is fine-tuned using chatgpt
// MARK: - ViewController Class Definition
class ViewController: UIViewController {
    // MARK: - UI Elements
    @IBOutlet weak var uploadButton: UIButton! // Button to upload a file
    @IBOutlet weak var resultTextView: UITextView! // TextView to display results or messages
    @IBOutlet weak var aiLabel: UILabel! // Label to display AI or Human detection results
    @IBOutlet weak var animationView: UIView! // View for displaying animation effects

    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI() // Initial setup for the user interface
    }

    // MARK: - Setup UI
    private func setupUI() {
        uploadButton.setTitle("Upload File", for: .normal) // Set the title of the upload button
        resultTextView.text = "Please click upload files to upload files..." // Set the default instruction text
        animationView.isHidden = true // Hide the animation view by default
        animationView.layer.cornerRadius = animationView.frame.width / 2 // Make the animation view circular
        animationView.backgroundColor = .green // Set the initial background color of the animation view
    }

    // MARK: - Actions
    @IBAction func uploadButtonTapped(_ sender: UIButton) {
        // Open the document picker to select audio or image files
        let documentPicker = UIDocumentPickerViewController(forOpeningContentTypes: [.audio, .image])
        documentPicker.delegate = self // Set the delegate for the picker
        documentPicker.allowsMultipleSelection = false // Restrict to single file selection
        present(documentPicker, animated: true, completion: nil) // Present the document picker
    }
    
    // MARK: - AI Detection Animation
    private func showAIDetectionAnimation(isAI: Bool) {
        // Create a ripple effect view as a subview of animationView
        let rippleView = UIImageView(frame: animationView.bounds)
        rippleView.image = UIImage(named: isAI ? "red_pattern" : "green_pattern") // Choose image based on AI detection
        rippleView.contentMode = .scaleAspectFill // Scale image to fill the view
        rippleView.clipsToBounds = true // Ensure image stays within bounds of rippleView
        animationView.addSubview(rippleView) // Add rippleView to animationView

        // Configure animation for ripple effect
        UIView.animateKeyframes(withDuration: 2.0, delay: 0, options: [.calculationModeLinear], animations: {
            UIView.addKeyframe(withRelativeStartTime: 0.0, relativeDuration: 1.0) {
                rippleView.transform = CGAffineTransform(scaleX: 10.0, y: 10.0) // Expand ripple effect
                rippleView.alpha = 0.0 // Fade out the ripple
            }
        }) { _ in
            rippleView.removeFromSuperview() // Remove ripple view after animation
        }

        // Update the animationView with image and transform
        animationView.isHidden = false // Unhide animation view
        let image = UIImage(named: isAI ? "red_pattern" : "green_pattern") // Set background image based on AI status
        animationView.layer.contents = image?.cgImage
        animationView.layer.contentsGravity = .resizeAspectFill // Adjust content to fill view
        animationView.layer.masksToBounds = true // Prevent content from exceeding bounds
        animationView.transform = CGAffineTransform(scaleX: 1.0, y: 1.0) // Reset transform if needed

        // Trigger haptic feedback for user
        triggerHapticFeedback(isAI: isAI)
    }

    // MARK: - Enhanced Haptic Feedback
    private func triggerHapticFeedback(isAI: Bool) {
        // Create a haptic feedback generator and trigger it based on AI detection result
        let generator = UINotificationFeedbackGenerator()
        generator.notificationOccurred(isAI ? .error : .success) // Error for AI, success for Human
    }

    // MARK: - Networking
    private func classifyFile(fileURL: URL) {
        let fileName = fileURL.lastPathComponent // Get the name of the file
        // Call the API to upload and predict the file's classification
        BioAuthAPI.shared.uploadAndPredict(fileURL: fileURL) { [weak self] result in
            DispatchQueue.main.async { // Ensure updates happen on the main thread
                switch result {
                case .success(let response):
                    // Display prediction and probability in the resultTextView
                    self?.resultTextView.text = """
                    Prediction: \(response.prediction)
                    Probability: \(response.probability)
                    """
                    if response.prediction == "real" {
                        // Update UI for a real human voice
                        self?.aiLabel.text = "Result: \(fileName) is Human Voice 👦"
                        self?.showAIDetectionAnimation(isAI: false)
                    } else {
                        // Update UI for an AI-generated voice
                        self?.aiLabel.text = "Result: \(fileName) is AI Voice 🤖"
                        self?.showAIDetectionAnimation(isAI: true)
                    }
                case .failure(let error):
                    // Display error message
                    self?.resultTextView.text = "Error: \(error.localizedDescription)"
                    self?.aiLabel.text = "AI Voice Synthesis: Unknown"
                }
            }
        }
    }
}

// MARK: - UIDocumentPickerDelegate
extension ViewController: UIDocumentPickerDelegate {
    func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
        guard let fileURL = urls.first else { return } // Ensure a file was selected
        
        let fileName = fileURL.lastPathComponent // Get the file name
        let fileExtension = fileURL.pathExtension.lowercased() // Get the file extension in lowercase

        if fileExtension != "mp3" {
            // Show an alert if the selected file is not an MP3
            showAlert(title: "Unsupported File", message: "Only MP3 files are supported. Please select an MP3 file.")
            return
        }
        
        aiLabel.text = "Selected File: \(fileName)" // Update the label with the file name
        let canAccess = fileURL.startAccessingSecurityScopedResource() // Start accessing the file
        defer {
            if canAccess {
                fileURL.stopAccessingSecurityScopedResource() // Stop accessing the file when done
            }
        }
        classifyFile(fileURL: fileURL) // Send the file for classification
    }

    func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) {
        print("Document picker was cancelled.") // Log cancellation
    }
    
    private func showAlert(title: String, message: String) {
        // Show an alert with a given title and message
        let alert = UIAlertController(title: title, message: message, preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default, handler: nil)) // Add an OK button
        present(alert, animated: true, completion: nil) // Present the alert
    }
}




