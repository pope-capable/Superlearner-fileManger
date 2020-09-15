const database = require("../src/models");
const Sequelize = require("sequelize");
const Op = Sequelize.Op;

class ProcessServives {
  // Create project Record
  static async createProcess(data) {
    try {
      return await database.processes.create(data);
    } catch (error) {
      throw error;
    }
  }

  // get one project breakdown
  static async getProcesses(data) {
    try {
      return await database.processes.findAll({
        where: { projectId: data.projectId }, include: [{model: database.files_mains}]
      });
    } catch (error) {
      throw error;
    }
  }

    // update processes
    static async updateProcess(data) {
      try {
        return await database.processes.update(data, {
          where: {id: data.processId }
        });
      } catch (error) {
        throw error;
      }
    }

}

module.exports = ProcessServives;
