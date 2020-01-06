/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include "helper_functions.h"
#define LIMIT 0.0001

using std::string;
using std::vector;
using std::default_random_engine;
using std::normal_distribution;
using std::numeric_limits;
using std::discrete_distribution;



void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  std::default_random_engine gen;
  
  num_particles = 100;  // TODO: Set the number of particles
  std:: normal_distribution<double> dist_x(x, std[0]);
  std:: normal_distribution<double> dist_y(y, std[1]);
  std:: normal_distribution<double> dist_theta(theta, std[2]);
  
  for (int i = 0; i < num_particles; ++i) 
  {    
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0; 
    particles.push_back(particle); 
    weights.push_back(particle.weight);
  }

  is_initialized = true; // particle filter initialized
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) 
{
  /**
   * TODO: Add measurements (prediction of motion) to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  
  std:: normal_distribution<double> dist_x(0.0, std_pos[0]);
  std:: normal_distribution<double> dist_y(0.0, std_pos[1]);
  std:: normal_distribution<double> dist_theta(0.0, std_pos[2]);
  
  for (int i = 0; i < num_particles; ++i) 
  {  
    // when yaw rate is larger than 0.0001
    if (fabs(yaw_rate) > LIMIT) 
    { 
      particles[i].x += (velocity/yaw_rate)* (sin(particles[i].theta + yaw_rate * delta_t)-sin(particles[i].theta)); 
      particles[i].y += (velocity/yaw_rate) * (cos(particles[i].theta)-cos(particles[i].theta + yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }
    // when yaw rate is smaller than 0.0001
    else
    { 
      particles[i].x += velocity * delta_t * cos(particles[i].theta); 
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
  // add random Gaussian noise in measurement of position.
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) 
{
  /**
   * TODO: Find the predicted measurement (particle) that is closest to each 
   *   observed measurement (either Lidar or Radar) and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
 for (unsigned int i = 0; i < observations.size(); ++i)
 {
   double min_dist = std::numeric_limits<double>::max();
   int id_in_map = -1; 
   for (unsigned int j = 0; j < predicted.size(); ++j)
   { 
      double rmse = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if (rmse < min_dist) 
      {
        min_dist = rmse;
        id_in_map = predicted[j].id; 
      }
      observations[i].id = id_in_map;
    }
  } 
}
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) 
{
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

// Transform the observations from car's coordinates to the map's coordinates 

  for (int i = 0; i < num_particles; ++i)
  {   
    Particle& star = particles[i];
    vector<LandmarkObs> TOBS;
    for (unsigned int j = 0; j < observations.size(); ++j)
    {
      double x_part = star.x;
      double y_part = star.y;
      double x_obs = observations[j].x;
      double y_obs = observations[j].y;
      double theta = star.theta; 
       // transform to map x and y coordinates
      double x_map = x_part + (cos(theta) * x_obs) - (sin(theta) * y_obs);
      double y_map = y_part + (sin(theta) * x_obs) + (cos(theta) * y_obs);
      TOBS.push_back(LandmarkObs{observations[j].id, x_map, y_map});
    }
  
  // Identify the map landmarks inside the sensor range 
    vector<LandmarkObs> predicted;
    for (unsigned int k = 0; k < map_landmarks.landmark_list.size(); ++k) 
    { 
      int    land_id = map_landmarks.landmark_list[k].id_i;
      float  land_x = map_landmarks.landmark_list[k].x_f;
      float  land_y = map_landmarks.landmark_list[k].y_f;
      double pred_distance = dist(star.x, star.y, land_x, land_y);
      if (pred_distance <= sensor_range) 
      {
        predicted.push_back(LandmarkObs{land_id, land_x, land_y});
      }
    }

    // Nearest Neighbor Data Association
    dataAssociation(predicted, TOBS);

    // Calcluate the weight of partciles 
    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;

    particles[i].weight = 1.0;
    double x_obs, y_obs, mu_x, mu_y;

    for (unsigned int l = 0; l < TOBS.size(); ++l)
    {
      x_obs =  TOBS[l].x;
      y_obs =  TOBS[l].y;
      for (unsigned int m = 0; m < predicted.size(); ++m)
      {
        if (predicted[m].id == TOBS[l].id)
        {
          mu_x = predicted[m].x;
          mu_y = predicted[m].y;
        }
      }
      // Landmark measurement uncertainty [x [m], y [m]]
      double sig_x = std_landmark[0];
      double sig_y = std_landmark[1];
      double gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);
      double exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2))) 
                      + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));

      double star_weight = gauss_norm * exp(-exponent); 
      
      particles[i].weight *= star_weight;
      
      // Append particle associations
      associations.push_back(TOBS[l].id);
      sense_x.push_back(x_obs);
      sense_y.push_back(y_obs);
    }
  
    weights[i] = star.weight;
    SetAssociations(star, associations, sense_x, sense_y);
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::default_random_engine gen;
  std::discrete_distribution<> dist_weighted(weights.begin(),weights.end());
  std::vector<Particle> particles_sampled;

  for (unsigned int i = 0; i < particles.size() ; ++i) 
  {
    int sample_index = dist_weighted(gen);
    particles_sampled.push_back(particles[sample_index]);
  }
  particles = particles_sampled;  
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}