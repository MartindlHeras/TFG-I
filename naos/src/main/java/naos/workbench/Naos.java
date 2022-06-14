package naos.workbench;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.SwingConstants;

public class Naos implements ActionListener {
	
	private static JFrame exitFrame = new JFrame();
	private static JPanel exitPanel = new JPanel();
//	private static JFrame frame;
//	private static JPanel panel;
	private static JLabel dataLabel;
	private static JLabel appLabel;
	private static JLabel mutomvoLabel;
	private static JLabel maloneLabel;
	private static JLabel coresLabel;
	private static JLabel exitLabel;
	
	private static JTextField dataText;
	private static JTextField appText;
	private static JTextField mutomvoText;
	private static JTextField maloneText;
	private static JTextField coresText;

	private static JButton fill;
//	private static JButton train;
	private static JButton mutate;
	private static JButton predict;
	private static JButton ok;
	
	public static void main(String[] args) {
		
		JFrame frame = new JFrame();
		JPanel panel = new JPanel();
		
		frame.setSize(500,500);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setLocationRelativeTo(frame);
		frame.add(panel);
		frame.setTitle("Naos");
		
		panel.setLayout(null);
		
		// DATA PATH
		dataLabel = new JLabel("New data path:");
		dataLabel.setBounds(50, 50, 125, 30);
		panel.add(dataLabel);
		
		dataText = new JTextField(20);
		dataText.setBounds(180, 50, 250, 30);
		panel.add(dataText);
		
		// FILL AND TRAIN
		fill = new JButton("Fill");
		fill.setBounds(170, 100, 160, 30);
		fill.addActionListener(new Naos());
		panel.add(fill);
		
//		train = new JButton("Train");
//		train.setBounds(250, 100, 160, 30);
//		train.addActionListener(new Naos());
//		panel.add(train);
		
		// APP NAME
		appLabel = new JLabel("App name:");
		appLabel.setBounds(50, 150, 125, 30);
		panel.add(appLabel);
		
		appText = new JTextField(20);
		appText.setBounds(200, 150, 250, 30);
		panel.add(appText);
		
		// MUTOMVO PATH
		mutomvoLabel = new JLabel("Mutomvo path:");
		mutomvoLabel.setBounds(50, 200, 125, 30);
		panel.add(mutomvoLabel);
		
		mutomvoText = new JTextField(20);
		mutomvoText.setBounds(200, 200, 250, 30);
		panel.add(mutomvoText);
		
		// MALONE PATH
		maloneLabel = new JLabel("Malone path:");
		maloneLabel.setBounds(50, 250, 125, 30);
		panel.add(maloneLabel);
		
		maloneText = new JTextField(20);
		maloneText.setBounds(200, 250, 250, 30);
		panel.add(maloneText);
		
		// NUMBER OF CORES
		coresLabel = new JLabel("Number of cores:");
		coresLabel.setBounds(50, 300, 140, 30);
		panel.add(coresLabel);
		
		coresText = new JTextField(20);
		coresText.setBounds(200, 300, 250, 30);
		panel.add(coresText);
		
		// MUTATE AND PREDICT
		mutate = new JButton("Mutate");
		mutate.setBounds(90, 350, 160, 30);
		mutate.addActionListener(new Naos());
		panel.add(mutate);
		
		predict = new JButton("Predict");
		predict.setBounds(250, 350, 160, 30);
		predict.addActionListener(new Naos());
		panel.add(predict);
		
		// LO DE DESPUES QUE A SABER QUE PONGO		
		exitFrame.setSize(300,300);
		exitFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		exitFrame.setLocationRelativeTo(panel);
		exitFrame.add(exitPanel);
		
		exitPanel.setLayout(null);
		
		// FILL AND TRAIN
		ok = new JButton("OK");
		ok.setBounds(100, 150, 100, 30);
		ok.addActionListener(new Naos());
		exitPanel.add(ok);
		
		exitLabel = new JLabel("Something went wrong.", SwingConstants.CENTER);
		exitLabel.setBounds(50, 50, 200, 100);
		exitPanel.add(exitLabel);
		
		frame.setVisible(true);
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		
		switch (e.getActionCommand()) {
		case "Fill":
			if (Trainer.fill(dataText.getText()) != null) {
				exitLabel.setText("DB filled successfully!");
			}
			else {
				exitLabel.setText("Something went wrong.");
			}
			exitFrame.setVisible(true);
			break;
		case "Train":
			exitLabel.setText(Trainer.train(dataText.getText()));
			exitFrame.setVisible(true);			
			break;
		case "Mutate":
			exitLabel.setText(Predictor.mutate(appText.getText(), mutomvoText.getText(), maloneText.getText()));
			exitFrame.setVisible(true);		
			break;
		case "Predict":
			exitLabel.setText(Predictor.predict(appText.getText(), mutomvoText.getText(), maloneText.getText(), coresText.getText()));
			exitFrame.setVisible(true);			
			break;
		case "OK":
			exitFrame.setVisible(false);
			break;
		}
	}
}
