var synaptic = require('synaptic');
var request = require('ajax-request');
var fs = require('fs');

/*
var googleMapsClient = require('@google/maps').createClient({
  key: 'AIzaSyB_uY1nM7Cjqa0Ezm42w-SYuDwhu8vQUR0'
});

googleMapsClient.geocode({
  address: '1600 Amphitheatre Parkway, Mountain View, CA'
}, function(err, response) {
  if (!err) {
    console.log(response.json.results);
  }
});*/

//neural network
var myNetwork = new synaptic.Architect.Perceptron(39, 20, 1);
var trainer = new synaptic.Trainer(myNetwork);

const trainingSetSize = 10;
var getNormalisedData = function(){
  //read in training data
  var obj = JSON.parse(fs.readFileSync('road.json'));

  //normalise data for the neural network
  var normalisedData = [];
  for(var x=0; x < trainingSetSize; x++){
    var day = [0,0,0,0,0,0,0]; day[obj[x].Day_of_Week -1] = 1;
    var lightCond = [0,0,0,0,0,0,0]; lightCond[obj[x].Light_Conditions -1] = 1;
    var weatherCond = [0,0,0,0,0,0,0,0,0]; weatherCond[obj[x].Weather_Conditions -1] = 1;
    var speed = [obj[x].Speed_limit/100];

    var roadTypeMap = [1,2,3,6,7,9], roadTypeIndex = roadTypeMap.indexOf(obj[x].Road_Type);
    var roadType = [0,0,0,0,0,0]; roadType[roadTypeIndex] = 1;

    var juncTypeMap = [0,1,2,3,5,6,7,8,9], juncTypeIndex = juncTypeMap.indexOf(obj[x].Junction_Detail);
    var juncType = [0,0,0,0,0,0,0,0,0]; juncType[juncTypeIndex] = 1;

    var tmp = {
      input: day.concat(lightCond).concat(weatherCond).concat(speed).concat(roadType).concat(juncType),
      output: [(obj[x].Accident_Severity < 3)?1:0]
    };
    normalisedData.push(tmp);
  }
  return normalisedData;
};

var ndata = getNormalisedData();
trainer.train(ndata, {
	rate: .1,
	iterations: 20000,
	shuffle: true,
	log: 1000
});

console.log("probability", myNetwork.activate(ndata[0].input));
