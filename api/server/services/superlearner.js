const database = require("../src/models");
const Sequelize = require("sequelize");
const Op = Sequelize.Op;

class SuperlearnerServe {

  // Create project Record
  static async createSuperlearner(data) {
    try {
      return await database.superlearners.create(data);
    } catch (error) {
      throw error;
    }
  }

  // get one project breakdown
  static async getSuperLearners(data) {
    try {
      return await database.superlearners.findAll({
        where: { projectId: data.projectId },
      });
    } catch (error) {
      throw error;
    }
  }



}

module.exports = SuperlearnerServe;
