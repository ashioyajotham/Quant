@Controller
public class PredictionController {
    @GetMapping("/")
    public String showHome() {
        return "home";
    }

    @PostMapping("/predict")
    public String predict(@RequestParam("feature1") double feature1,
                          @RequestParam("feature2") double feature2,
                          @RequestParam("feature3") double feature3,
                          @RequestParam("feature4") double feature4,
                          @RequestParam("feature5") double feature5,
                          @RequestParam("feature6") double feature6,
                          Model model) {
        // perform prediction logic here
        double prediction = 0.0;

        // add prediction value to the model
        model.addAttribute("prediction", prediction);

        return "result";
    }
