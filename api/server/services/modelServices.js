const database = require("../src/models");
const Sequelize = require("sequelize");
const Op = Sequelize.Op;

class ModelServices {
  // Create project Record
  static async createModel(data) {
    try {
      return await database.models.create(data);
    } catch (error) {
      throw error;
    }
  }

  // get one project breakdown
  static async getModels(data) {
    try {
      return await database.models.findAll({
        where: { projectId: data.projectId },
      });
    } catch (error) {
      throw error;
    }
  }

    // update models
    static async updateModel(data) {
      try {
        return await database.models.update(data, {
          where: {id: data.processId }
        });
      } catch (error) {
        throw error;
      }
    }

}

module.exports = ModelServices;
